"""FPGA-scoped display localization; acquisition and saved values remain canonical.

The GUI was built with runtime-generated Korean status text. This adapter
retains each object's source text and retranslates new status updates without
touching editable values, item data, selected indexes, workers or device APIs.
"""
import re
from pathlib import Path
import json
from functools import lru_cache

from PyQt5.QtCore import QObject, QTimer, QEvent, QSignalBlocker, Qt
from PyQt5.QtWidgets import (QApplication,QWidget,QLabel,QAbstractButton,QGroupBox,
    QComboBox,QTabWidget,QListWidget,QTableWidget,QMessageBox,QProgressBar,QLineEdit,
    QFileDialog as QtFileDialog)
import pyqtgraph as pg
from fpga_i18n_catalog import TRANSLATIONS
from session_storage import atomic_json


HANGUL=re.compile('[가-힣]')
WINDOWS_PATH=re.compile(r'((?:[A-Za-z]:[\\/]|\\\\)[^\r\n<>]*)')


class DisplayCatalog:
    def __init__(self):
        keys=sorted(TRANSLATIONS,key=len,reverse=True)
        self.pattern=re.compile('|'.join(
            ('(?<![가-힣])' if HANGUL.match(k[0]) else '')+re.escape(k)+
            ('(?![가-힣])' if HANGUL.fullmatch(k[-1]) else '') for k in keys))

    @lru_cache(maxsize=8192)
    def english(self,text):
        if not HANGUL.search(str(text)): return str(text)
        return ''.join(part if i%2 else self.pattern.sub(lambda m:TRANSLATIONS[m.group()],part)
                       for i,part in enumerate(WINDOWS_PATH.split(str(text))))


def source_widget_text(widget):
    """Canonical label/list-item text for legacy code; never use on user input."""
    current=widget.text()
    if isinstance(widget,QObject):
        saved=widget.property('_fpga_l10n_text')
    else:
        saved=widget.data(Qt.UserRole+843)
    return saved[0] if saved and current==saved[1] else current


def source_combo_text(widget):
    return source_combo_item(widget,widget.currentIndex())


def source_combo_item(widget,index):
    saved=widget.property('_fpga_l10n_combo') or []
    if 0<=index<len(saved) and saved[index] and widget.itemText(index)==saved[index][1]:
        return saved[index][0]
    return widget.itemText(index) if index>=0 else ''


class FileDialog:
    """Translate app captions; native dialog controls follow the OS language."""
    @staticmethod
    def _call(name,*args,**kwargs):
        args=list(args)
        parent=args[0] if args else kwargs.get('parent')
        manager=getattr(parent,'language_manager',None)
        if manager:
            if len(args)>1: args[1]=manager.text(args[1])
            elif 'caption' in kwargs: kwargs['caption']=manager.text(kwargs['caption'])
        return getattr(QtFileDialog,name)(*args,**kwargs)

    @staticmethod
    def getSaveFileName(*args,**kwargs): return FileDialog._call('getSaveFileName',*args,**kwargs)
    @staticmethod
    def getOpenFileName(*args,**kwargs): return FileDialog._call('getOpenFileName',*args,**kwargs)
    @staticmethod
    def getOpenFileNames(*args,**kwargs): return FileDialog._call('getOpenFileNames',*args,**kwargs)
    @staticmethod
    def getExistingDirectory(*args,**kwargs): return FileDialog._call('getExistingDirectory',*args,**kwargs)


class FPGALanguageManager(QObject):
    def __init__(self,window,state_file):
        super().__init__(window)
        self.window=window
        self.state_file=Path(state_file)
        self.catalog=DisplayCatalog()
        self.language='ko'
        try:
            language=json.loads(self.state_file.read_text(encoding='utf-8')).get('language','ko')
            if language in ('ko','en'): self.language=language
        except (OSError,ValueError,TypeError): pass
        self.updating=False
        self.pending=False
        self.active=True
        self.timer=QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.refresh)
        QApplication.instance().installEventFilter(self)
        self.timer.start()

    def text(self,source):
        return self.catalog.english(source) if self.language=='en' else str(source)

    def set_language(self,language):
        if language not in ('ko','en'): raise ValueError('Unsupported language')
        self.language=language
        self.refresh()
        atomic_json(self.state_file,dict(version=1,language=language))

    def _owned(self,widget):
        while isinstance(widget,QWidget):
            if widget is self.window: return True
            widget=widget.parentWidget()
        return False

    def eventFilter(self,obj,event):
        if event.type() in (QEvent.Show,QEvent.ToolTip) and isinstance(obj,QWidget) and self._owned(obj):
            # Avoid changing child layouts during native window creation.
            if not self.pending:
                self.pending=True
                QTimer.singleShot(0,self._deferred_refresh)
        return False

    def _deferred_refresh(self):
        self.pending=False
        self.refresh()

    def _field(self,obj,key,getter,setter):
        current=getter()
        if not isinstance(current,str) or not current: return
        prop='_fpga_l10n_'+key
        previous=obj.property(prop)
        source=previous[0] if previous and current==previous[1] else current
        rendered=self.text(source)
        obj.setProperty(prop,[source,rendered])
        if current!=rendered: setter(rendered)

    def _item(self,item):
        if item is None: return
        current=item.text()
        previous=item.data(Qt.UserRole+843)
        source=previous[0] if previous and current==previous[1] else current
        rendered=self.text(source)
        item.setData(Qt.UserRole+843,[source,rendered])
        if current!=rendered: item.setText(rendered)

    def _widgets(self,root):
        for obj in [root]+root.findChildren(QWidget):
            if obj.property('fpgaLanguageSelector'): continue
            self._field(obj,'tooltip',obj.toolTip,obj.setToolTip)
            self._field(obj,'window',obj.windowTitle,obj.setWindowTitle)
            if isinstance(obj,(QLabel,QAbstractButton)):
                self._field(obj,'text',obj.text,obj.setText)
            if isinstance(obj,QGroupBox): self._field(obj,'title',obj.title,obj.setTitle)
            if isinstance(obj,QProgressBar): self._field(obj,'format',obj.format,obj.setFormat)
            if isinstance(obj,QLineEdit): self._field(obj,'placeholder',obj.placeholderText,obj.setPlaceholderText)
            if isinstance(obj,QMessageBox):
                self._field(obj,'message',obj.text,obj.setText)
                self._field(obj,'informative',obj.informativeText,obj.setInformativeText)
                self._field(obj,'detailed',obj.detailedText,obj.setDetailedText)
            if isinstance(obj,QTabWidget):
                for i in range(obj.count()):
                    self._field(obj,'tab_'+str(i),lambda i=i:obj.tabText(i),lambda text,i=i:obj.setTabText(i,text))
            if isinstance(obj,QComboBox):
                blocker=QSignalBlocker(obj)
                old=obj.property('_fpga_l10n_combo') or []
                values=[]
                for i in range(obj.count()):
                    current=obj.itemText(i)
                    source=old[i][0] if i<len(old) and current==old[i][1] else current
                    translated=self.text(source)
                    values.append([source,translated])
                    if current!=translated: obj.setItemText(i,translated)
                obj.setProperty('_fpga_l10n_combo',values)
                del blocker
            if isinstance(obj,QListWidget):
                blocker=QSignalBlocker(obj)
                for i in range(obj.count()): self._item(obj.item(i))
                del blocker
            if isinstance(obj,QTableWidget):
                blocker=QSignalBlocker(obj)
                for i in range(obj.columnCount()): self._item(obj.horizontalHeaderItem(i))
                for row in range(obj.rowCount()):
                    for col in range(obj.columnCount()): self._item(obj.item(row,col))
                del blocker
            # Never translate QLineEdit contents or numerical values.
            if hasattr(obj,'suffix') and hasattr(obj,'setSuffix'):
                self._field(obj,'suffix',obj.suffix,obj.setSuffix)

    def _plots(self):
        scenes=set()
        for widget in self.window.findChildren(QWidget):
            if isinstance(widget,(pg.PlotWidget,pg.GraphicsLayoutWidget)):
                scenes.add(widget.scene())
        for scene in scenes:
            for item in scene.items():
                if isinstance(item,pg.LabelItem):
                    self._field(item,'label',lambda:item.text,lambda text:item.setText(text,**item.opts))

    def refresh(self,root=None):
        if self.updating or not self.active: return
        self.updating=True
        try:
            self._widgets(root or self.window)
            if root is None:
                self._plots()
                for popup in QApplication.topLevelWidgets():
                    if popup is not self.window and self._owned(popup): self._widgets(popup)
        finally:
            self.updating=False

    def shutdown(self):
        self.active=False
        self.timer.stop()
        QApplication.instance().removeEventFilter(self)


def install_language_selector(window,state_file):
    window.language_manager=FPGALanguageManager(window,state_file)
    combo=QComboBox()
    combo.setObjectName('fpga_language_selector')
    combo.setProperty('fpgaLanguageSelector',True)
    combo.addItem('한국어','ko')
    combo.addItem('English','en')
    combo.setCurrentIndex(combo.findData(window.language_manager.language))
    combo.currentIndexChanged.connect(lambda _index:window.language_manager.set_language(combo.currentData()))
    window.language_combo=combo
    label=QLabel('언어 / Language')
    label.setProperty('fpgaLanguageSelector',True)
    layout=window.status_label.parentWidget().layout()
    layout.insertWidget(0,combo)
    layout.insertWidget(0,label)
    window.language_manager.refresh()
