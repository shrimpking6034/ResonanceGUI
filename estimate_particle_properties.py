"""Add buoyant-mass and density estimates to an exported offline-review folder."""
import argparse
import json
from pathlib import Path

import pandas as pd

from particle_review_core import ParticlePropertySettings, estimate_particle_properties


def estimate_folder(folder, settings, input_name=None):
    folder=Path(folder).resolve()
    candidates=[folder/input_name] if input_name else [folder/'exported_events.csv',folder/'retained_review.csv']
    source=next((path for path in candidates if path and path.exists()),None)
    if source is None:
        raise FileNotFoundError('폴더에 exported_events.csv 또는 retained_review.csv가 없습니다.')
    table=pd.read_csv(source,encoding='utf-8-sig')
    rows,calibration=estimate_particle_properties(table.to_dict('records'),settings)
    target=folder/'particle_properties.csv'
    pd.DataFrame(rows).to_csv(target,index=False,encoding='utf-8-sig')
    pair_target=folder/'round_trip_pairs.csv'
    pd.DataFrame(calibration['pair_rows']).to_csv(pair_target,index=False,encoding='utf-8-sig')
    summary=dict(source=str(source),output=str(target),settings=settings.__dict__,calibration=calibration,
                 pair_output=str(pair_target),
                 note=('개별 통과값은 품질 확인용이며 질량은 왕복쌍 대표 신호에 한 번만 적용합니다. '
                       'reference 보정은 알려진 기준 입자의 지름·밀도로 감도를 정하므로 같은 집단의 질량·밀도·지름을 독립 검증하지 않습니다. '
                       '미지 시료에는 기준 측정에서 얻은 sensitivity_pg_hz를 직접 감도 방식으로 적용하세요.'))
    (folder/'particle_properties.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return target,calibration


def main():
    parser=argparse.ArgumentParser(description='오프라인 입자 후보의 부유질량·밀도 추정')
    parser.add_argument('folder',help='particle_graphs_* 결과 폴더')
    parser.add_argument('--mode',type=int,default=6)
    parser.add_argument('--diameter-min-um',type=float,default=190.)
    parser.add_argument('--diameter-max-um',type=float,default=220.)
    parser.add_argument('--fluid-density',type=float,default=997.,help='kg/m^3')
    parser.add_argument('--reference-density',type=float,default=7930.,help='kg/m^3')
    parser.add_argument('--sensitivity-pg-hz',type=float,help='지정하면 기준 입자 대신 직접 감도 보정 사용')
    parser.add_argument('--representative',choices=['mean','median'],default='mean',help='왕복쌍 평균들을 모으는 대표값 방식')
    parser.add_argument('--infer',choices=['mass_only','density_from_diameter','diameter_from_density'],default='mass_only',help='외부에서 알고 있는 물성에 따른 추가 역산')
    args=parser.parse_args()
    method='sensitivity' if args.sensitivity_pg_hz is not None else 'reference'
    settings=ParticlePropertySettings(mode_number=args.mode,diameter_min_um=args.diameter_min_um,
        diameter_max_um=args.diameter_max_um,fluid_density_kg_m3=args.fluid_density,
        calibration_method=method,reference_density_kg_m3=args.reference_density,
        sensitivity_pg_hz=args.sensitivity_pg_hz or 1.,representative_method=args.representative,
        inference_method=args.infer)
    target,calibration=estimate_folder(args.folder,settings)
    print(f'저장: {target}')
    print(f"질량 감도: {calibration['sensitivity_pg_hz']:,.3f} pg/Hz")
    print(f"완전한 왕복쌍: {calibration['complete_pair_count']}개")
    print(f"대표 신호: {calibration['representative_depth_hz']:.6g} Hz")
    print(f"대표 부유질량: {calibration['representative_buoyant_mass_pg']:.6g} pg")


if __name__=='__main__':main()
