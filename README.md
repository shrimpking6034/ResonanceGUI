# ResonanceGUI

공진기 주파수 스윕, 공진 피팅, PLL 추적 및 입자 신호 분석을 위한 Python 데스크톱 애플리케이션입니다. Zurich Instruments HF2LI와 SMR FPGA 기반 측정 환경을 지원합니다.

A Python desktop application for resonator frequency sweeps, resonance fitting, PLL tracking, and particle signal analysis. It supports measurement setups based on the Zurich Instruments HF2LI and an SMR FPGA controller.

[한국어](#한국어) · [English](#english)

## 한국어

### 주요 기능

- **공진 측정:** 주파수 스윕, 진폭·위상 표시, 공진 피크 검출, 공진 주파수 및 Q 추정.
- **PLL 및 실시간 기록:** PLL 파라미터 설정, 주파수 추적, 실시간 Plotter 및 CSV 내보내기.
- **HF2LI 측정:** LabOne C API를 통한 HF2LI 제어, NI-DAQ를 이용한 공압 제어 및 압력 모니터링.
- **FPGA 측정:** TCP 레지스터 제어, UDP 스트림 수신, 자동 모드 스윕, 초기 공진 탐색 및 PLL 보정 도구.
- **오프라인 분석:** 기준 잡음·드리프트·Allan 편차 분석, 입자 이벤트 검출·검토, 스윕 결과 비교 및 그래프 내보내기.
- **언어 선택:** FPGA 메인 GUI에서 한국어·영어 전환 지원.

### 설치

HF2LI 연동 코드는 Windows 64비트 LabOne DLL 경로를 사용합니다. 실제 장비 측정은 Windows 64비트 환경을 기준으로 준비하세요. 소스의 타입 문법을 위해 Python 3.10 이상을 사용하세요. 의존성 버전은 현재 저장소에서 고정하지 않습니다.

프로젝트 폴더에서 PowerShell을 열고 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy scipy pandas PyQt5 pyqtgraph matplotlib
```

NI-DAQ 기능을 사용할 경우 Python 패키지와 별도로 NI-DAQmx 드라이버도 설치해야 합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install nidaqmx
```

HF2LI에는 LabOne 설치와 실행 중인 HF2 Data Server가 필요합니다. 기본 서버 주소는 `localhost:8005`입니다. `zi_hf2_api.py`는 `ctypes`로 네이티브 C API를 호출하며, 기본 DLL 경로는 다음과 같습니다.

```text
C:\Program Files\Zurich Instruments\LabOne\API\C\lib\ziAPI-win64.dll
```

설치 위치가 다르면 `zi_hf2_api.py`의 `DLL_PATH`를 확인하세요. 이 프로젝트의 HF2LI 연결은 `zhinst` Python 패키지를 사용하지 않습니다.

### 실행

아래 명령은 프로젝트 폴더에서 실행합니다.

| 용도 | 명령 | Windows 실행 파일 |
| --- | --- | --- |
| HF2LI 메인 GUI | `.\.venv\Scripts\python.exe resonance_gui.py` | `Run_HF2LI_ResonanceGUI.bat` |
| FPGA 메인 GUI | `.\.venv\Scripts\python.exe resonance_fpga_gui.py` | `Run_FPGA_ResonanceGUI.bat` |
| FPGA 전용 GUI 및 시뮬레이터 | `.\.venv\Scripts\python.exe fpga_resonance_gui.py` | — |
| 기준 잡음 오프라인 분석 | `.\.venv\Scripts\python.exe baseline_offline_gui.py` | `Run_Baseline_Offline.bat` |
| 입자 신호 검토 | `.\.venv\Scripts\python.exe particle_review_gui.py` | `particle_review.pyw` |
| 스윕 비교 | `.\.venv\Scripts\python.exe sweep_comparison_gui.py` | `Run_Sweep_Comparison.bat` |

`.bat` 파일은 PATH의 `python`을 사용합니다. 가상환경을 사용하려면 먼저 활성화하거나 위의 명시적 경로 명령을 사용하세요. 장비 없이 FPGA 동작 흐름을 살펴보려면 `fpga_resonance_gui.py`의 `Simulator` 모드를 사용합니다.

저장된 CSV를 바로 여는 예시:

```powershell
.\.venv\Scripts\python.exe baseline_offline_gui.py "path\to\recording.csv"
.\.venv\Scripts\python.exe particle_review_gui.py "path\to\recording.csv"
```

### 장비 연결 및 측정 순서

1. 사용할 장비의 드라이버·서버 또는 네트워크 연결을 준비합니다.
2. 해당 메인 GUI를 실행하고 장비, 채널 및 출력 설정을 확인합니다.
3. 스윕으로 공진 피크를 찾고 피팅 결과의 공진 주파수와 Q를 확인합니다.
4. PLL 파라미터를 설정한 후 추적 상태를 확인하고 Plotter에서 데이터를 기록합니다.
5. 데이터를 저장한 뒤 오프라인 도구에서 잡음·입자 이벤트·스윕 결과를 분석합니다.

HF2LI 장치 검색과 별개로 NI-DAQ의 기본 별칭은 `Dev1`이며, `resonance_gui.py`의 `AUTO_DETECT_DAQ`는 기본적으로 `False`입니다. 다른 PC에서는 NI-MAX의 별칭과 채널 배선을 확인하세요. HF2LI 구동 출력 상한은 코드에서 10 Vpk, 공압 제어 출력은 0–10 V로 제한되며, 실제 사용값은 연결된 장비의 허용 범위에 맞춰 설정해야 합니다.

FPGA 기본 네트워크 설정은 `fpga_transport.py`의 `NetworkProfile`에 정의되어 있습니다.

| 항목 | 기본값 |
| --- | --- |
| FPGA 주소 | `192.168.100.2` |
| PC 인터페이스 주소 | `192.168.100.1` |
| TCP 포트 | `30` |
| UDP 멀티캐스트 그룹 | `224.1.1.1` |
| UDP 포트 | `5007` |

PC 네트워크 인터페이스 및 방화벽 설정을 측정 환경과 맞추세요. FPGA TCP 서버는 단일 클라이언트를 사용하므로 다른 제어 프로그램의 연결을 종료한 뒤 접속합니다.

### 데이터와 저장 위치

- Plotter CSV에는 `Time_s`, `Frequency_Hz`와 진폭·위상·PLL·압력 등의 열이 포함됩니다. 일부 열은 측정 장치에 따라 값이 없을 수 있습니다. 파일 앞부분에는 메타데이터 행이 들어갈 수 있습니다.
- 기준 잡음 및 입자 분석의 핵심 입력 열은 `Time_s`와 `Frequency_Hz`입니다. 기준 잡음 분석에는 중복 없이 증가하는 시간값과 유한한 수치가 필요합니다.
- 스윕 및 분석 도구는 기능에 따라 CSV, NPZ, JSON과 그래프 파일을 저장합니다. 저장 위치는 각 도구의 내보내기 기능에서 선택합니다.
- 소스 실행 시 GUI 상태는 `.resonance_state/`에 보관됩니다.
- 원시 측정 CSV와 `outputs/`, `diagnostics/` 등은 `.gitignore`로 제외됩니다. 실험 데이터는 별도로 백업하세요.

### 코드 구성 및 검증

| 파일 또는 그룹 | 역할 |
| --- | --- |
| `resonance_gui.py`, `zi_hf2_api.py`, `hf2_features.py` | HF2LI GUI 및 장비 연동 |
| `resonance_fpga_gui.py`, `fpga_*` | FPGA GUI, 제어, 스트리밍, 자동 스윕 및 보정 |
| `resonance_fit.py`, `sweep_math.py`, `pll_design.py`, `beam_modes.py` | 공진 피팅, 스윕 계산, PLL 설계 및 빔 모드 계산 |
| `baseline_analysis.py`, `baseline_offline*`, `experiment_analysis.py` | 잡음 및 실험 분석 |
| `particle_review*` | 입자 이벤트 검출, 검토 및 내보내기 |
| `sweep_comparison*` | 저장된 스윕 비교 및 그래프 생성 |
| `session_storage.py`, `allan_export.py` | 세션 저장 및 Allan 결과 내보내기 |
| `test_*.py` | 자동화 테스트 |

장비 연결 없이 실행하는 단위 테스트 예시:

```powershell
.\.venv\Scripts\python.exe -m unittest test_resonance_models test_baseline_analysis test_fpga_stream test_fpga_transport
```

`fpga_live_*.py`와 장비 진단 스크립트는 실제 장비를 대상으로 할 수 있습니다. 실행 전 해당 스크립트의 연결·출력 동작을 확인하세요.

## English

### Features

- **Resonance measurements:** frequency sweeps, amplitude and phase plots, peak detection, resonance frequency and Q estimation.
- **PLL and live recording:** PLL parameter control, frequency tracking, live plotting, and CSV export.
- **HF2LI integration:** control through the LabOne C API, with NI-DAQ pneumatic control and pressure monitoring.
- **FPGA integration:** TCP register control, UDP streaming, automatic mode sweeps, initial resonance search, and PLL refinement tools.
- **Offline analysis:** baseline noise, drift and Allan deviation analysis, particle event detection and review, sweep comparison, and graph export.
- **Language selection:** Korean and English in the FPGA main GUI.

### Installation

Prepare a 64-bit Windows environment for hardware measurements: the HF2LI integration uses a Windows 64-bit LabOne DLL. Use Python 3.10 or later for the source syntax. Dependency versions are currently unpinned.

Open PowerShell in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy scipy pandas PyQt5 pyqtgraph matplotlib
```

For NI-DAQ features, install the NI-DAQmx driver as well as its Python package:

```powershell
.\.venv\Scripts\python.exe -m pip install nidaqmx
```

HF2LI measurements require LabOne and a running HF2 Data Server, using `localhost:8005` by default. The `ctypes` wrapper in `zi_hf2_api.py` loads the native C API from:

```text
C:\Program Files\Zurich Instruments\LabOne\API\C\lib\ziAPI-win64.dll
```

Check `DLL_PATH` in that module if LabOne is installed elsewhere. This project's HF2LI connection does not use the `zhinst` Python package.

### Running the tools

Run these commands from the project directory:

| Tool | Command | Windows launcher |
| --- | --- | --- |
| HF2LI main GUI | `.\.venv\Scripts\python.exe resonance_gui.py` | `Run_HF2LI_ResonanceGUI.bat` |
| FPGA main GUI | `.\.venv\Scripts\python.exe resonance_fpga_gui.py` | `Run_FPGA_ResonanceGUI.bat` |
| Dedicated FPGA GUI and simulator | `.\.venv\Scripts\python.exe fpga_resonance_gui.py` | — |
| Offline baseline analysis | `.\.venv\Scripts\python.exe baseline_offline_gui.py` | `Run_Baseline_Offline.bat` |
| Particle signal review | `.\.venv\Scripts\python.exe particle_review_gui.py` | `particle_review.pyw` |
| Sweep comparison | `.\.venv\Scripts\python.exe sweep_comparison_gui.py` | `Run_Sweep_Comparison.bat` |

Batch launchers use `python` from PATH. Activate the virtual environment first, or use the explicit commands above. To explore the FPGA workflow without hardware, use `Simulator` mode in `fpga_resonance_gui.py`.

The baseline and particle review GUIs also accept a CSV path:

```powershell
.\.venv\Scripts\python.exe baseline_offline_gui.py "path\to\recording.csv"
.\.venv\Scripts\python.exe particle_review_gui.py "path\to\recording.csv"
```

### Hardware setup and workflow

1. Prepare the required drivers, server, or network connection.
2. Open the appropriate main GUI and verify the device, channels, and output settings.
3. Run a sweep, select a resonance peak, and check the fitted resonance frequency and Q.
4. Configure the PLL, check tracking behavior, and record data with the Plotter.
5. Export the recording and use the offline tools to analyze noise, particle events, or sweep results.

The NI-DAQ alias defaults to `Dev1`, with `AUTO_DETECT_DAQ = False` in `resonance_gui.py`, independently of HF2LI discovery. Check NI-MAX aliases and wiring when moving to another PC. The code limits HF2LI drive output to 10 Vpk and pneumatic control output to 0–10 V; choose operating values appropriate for the connected equipment.

Default FPGA networking is defined by `NetworkProfile` in `fpga_transport.py`:

| Setting | Default |
| --- | --- |
| FPGA address | `192.168.100.2` |
| PC interface address | `192.168.100.1` |
| TCP port | `30` |
| UDP multicast group | `224.1.1.1` |
| UDP port | `5007` |

Match the PC network interface and firewall settings to the measurement setup. The FPGA TCP server accepts one client, so disconnect other control software before connecting.

### Data and storage

- Plotter CSV exports contain `Time_s`, `Frequency_Hz`, and amplitude, phase, PLL, and pressure fields. Some fields may be unavailable for a given device. Metadata rows may precede the column header.
- The core input columns for baseline and particle analysis are `Time_s` and `Frequency_Hz`. Baseline analysis requires strictly increasing timestamps and finite numeric values.
- Sweep and analysis tools export CSV, NPZ, JSON, and graph files depending on the feature. Choose the destination through each tool's export controls.
- GUI state is stored in `.resonance_state/` when running from source.
- Raw measurement CSV files and folders such as `outputs/` and `diagnostics/` are excluded by `.gitignore`. Back up experimental data separately.

### Source layout and validation

| Files | Purpose |
| --- | --- |
| `resonance_gui.py`, `zi_hf2_api.py`, `hf2_features.py` | HF2LI GUI and device integration |
| `resonance_fpga_gui.py`, `fpga_*` | FPGA GUI, control, streaming, automatic sweeps, and refinement |
| `resonance_fit.py`, `sweep_math.py`, `pll_design.py`, `beam_modes.py` | Resonance fitting, sweep calculations, PLL design, and beam modes |
| `baseline_analysis.py`, `baseline_offline*`, `experiment_analysis.py` | Noise and experiment analysis |
| `particle_review*` | Particle event detection, review, and export |
| `sweep_comparison*` | Saved sweep comparison and graph generation |
| `session_storage.py`, `allan_export.py` | Session storage and Allan result export |
| `test_*.py` | Automated tests |

Example unit tests that do not require connected hardware:

```powershell
.\.venv\Scripts\python.exe -m unittest test_resonance_models test_baseline_analysis test_fpga_stream test_fpga_transport
```

The `fpga_live_*.py` scripts and device diagnostics may interact with real hardware. Check their connection and output behavior before running them.
