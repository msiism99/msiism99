from pathlib import Path
from datetime import datetime
import math
import re

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd


# =========================================================
# 1. 사용자 설정
# =========================================================
INPUT_DIR = Path.cwd() / 'WAFER_MAP_INPUT'
OUTPUT_ROOT_DIR = Path.cwd() / 'WAFER_MAP_OUTPUT'

# None이면 INPUT_DIR의 모든 *OVL_UFI_CHIP_ALL*.pkl을 읽어서 합칩니다.
# 파일 하나만 직접 지정할 때 사용합니다.
# 예: INPUT_PICKLE = INPUT_DIR / '2632__OVL_UFI_CHIP_ALL.pkl'
INPUT_PICKLE = None

# 비어 있으면 전체를 사용합니다. 원하는 값만 남기려면 목록에 입력합니다.
TARGET_UFI_IDS = ['UFI1019', 'UFI1027']
TARGET_ROOT_LOTS = []
TARGET_WAFERS = []
TARGET_PROCESSES = []
TARGET_STEPS = []
TARGET_PPIDS = []

# 현재 5단계 파일의 chip global coordinate는 mm 기준입니다.
# 과거 파일이 um라면 0.001로 변경합니다.
POSITION_SCALE_TO_MM = 1.0
WAFER_RADIUS_MM = 150.0

# 전체 선택 데이터의 1~99 percentile로 color scale을 통일합니다.
ROBUST_COLOR_PERCENTILES = (3.0, 97.0)

# metric별 수동 scale. 필요할 때만 값을 넣습니다.
# signed metric: (-limit, +limit), positive metric: (minimum, maximum)
MANUAL_COLOR_LIMITS = {
    # 'dx_absmax': (0.0, 3.0),
    # 'dy_absmax': (0.0, 3.0),
    # 'radial_ovl_max': (0.0, 4.0),
    # 'ufi_rate_pct': (0.0, 20.0),
}

SAVE_COMPARISON_PANEL = True
SAVE_SINGLE_MAPS = True
PREVIEW_FIRST_COMPARISON = True
FIGURE_DPI = 170

# Shot 경계 표시
SHOW_SHOT_BOUNDARIES = True
SHOT_BOUNDARY_COLOR = '#202020'
SHOT_BOUNDARY_LINEWIDTH = 0.85
SHOT_BOUNDARY_ALPHA = 0.90

# 여러 wafer를 합친 map을 나누는 기준입니다.
# root_lot_id와 wafer_id는 의도적으로 제외합니다.
MAP_GROUP_COLUMNS = [
    'selected_process_id',
    'selected_step_seq',
    'selected_ppid_main_ovl',
    'ufi_item_id',
]

METRIC_CONFIG = {
    'dx_absmax': {
        'title': 'OVL DX mean absmax',
        'unit': 'nm',
        'cmap': 'turbo',
        'signed': False,
        'aggregation': 'mean',
    },
    'dy_absmax': {
        'title': 'OVL DY mean absmax',
        'unit': 'nm',
        'cmap': 'turbo',
        'signed': False,
        'aggregation': 'mean',
    },
    'radial_ovl_max': {
        'title': 'OVL vector magnitude max',
        'unit': 'nm',
        'cmap': 'turbo',
        'signed': False,
        'aggregation': 'mean',
    },
    'ufi_rate_pct': {
        'title': 'UFI defect rate',
        'unit': '%',
        'cmap': 'YlOrRd',
        'signed': False,
    },
}

COMPARISON_METRICS = [
    'dx_absmax',
    'dy_absmax',
    'radial_ovl_max',
    'ufi_rate_pct',
]

SINGLE_MAP_METRICS = [
    'dx_absmax',
    'dy_absmax',
    'radial_ovl_max',
    'ufi_rate_pct',
]



def safe_token(value):
    token = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value).strip())
    return token.strip('._') or 'NA'


def normalize_filter_values(values, upper=True):
    result = []
    for value in values:
        text = str(value).strip()
        if text:
            result.append(text.upper() if upper else text)
    return sorted(set(result))


def find_input_pickles():
    if INPUT_PICKLE is not None:
        path = Path(INPUT_PICKLE)
        if not path.exists():
            raise FileNotFoundError(f'입력 PKL이 없습니다: {path}')
        return [path]

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f'입력 폴더가 없습니다: {INPUT_DIR}\n'
            'WAFER_MAP_INPUT 폴더를 만들고 OVL_UFI_CHIP_ALL PKL을 넣어주세요.'
        )

    candidates = sorted(INPUT_DIR.rglob('*OVL_UFI_CHIP_ALL*.pkl'))

    if not candidates:
        raise FileNotFoundError(
            f'OVL_UFI_CHIP_ALL PKL을 찾지 못했습니다: {INPUT_DIR}'
        )

    return candidates


INPUT_PATHS = find_input_pickles()
input_frames = []

for input_path in INPUT_PATHS:
    part = pd.read_pickle(input_path)
    part['_source_file'] = input_path.name
    input_frames.append(part)

raw = pd.concat(input_frames, ignore_index=True)

required_columns = set(
    [
        'root_lot_id',
        'wafer_id',
        *MAP_GROUP_COLUMNS,
    ]
    + [
        '_chip_center_global_x',
        '_chip_center_global_y',
        '_chip_pitch_x',
        '_chip_pitch_y',
        'shot_center_fcp_x',
        'shot_center_fcp_y',
    ]
    + [
        'dx_absmax',
        'dy_absmax',
        'radial_ovl_max',
        'ufi_event',
    ]
)
missing_columns = required_columns - set(raw.columns)

if missing_columns:
    raise KeyError(
        f'필수 컬럼이 없습니다: {sorted(missing_columns)}\n'
        'OVL_UFI_CHIP_ALL__*.pkl 파일인지 확인하세요.'
    )

data = raw.copy()

text_columns = [
    'root_lot_id',
    'wafer_id',
    'selected_process_id',
    'selected_step_seq',
    'selected_ppid_main_ovl',
    'ufi_item_id',
]

for column in text_columns:
    data[column] = data[column].astype('string').str.strip().str.upper()

for column in [
    '_chip_center_global_x',
    '_chip_center_global_y',
    '_chip_pitch_x',
    '_chip_pitch_y',
    'shot_center_fcp_x',
    'shot_center_fcp_y',
    'dx_absmax',
    'dy_absmax',
    'radial_ovl_max',
]:
    data[column] = pd.to_numeric(data[column], errors='coerce')

if pd.api.types.is_bool_dtype(data['ufi_event']):
    data['ufi_event'] = data['ufi_event'].astype('Int64')
else:
    event_numeric = pd.to_numeric(data['ufi_event'], errors='coerce')
    event_text = data['ufi_event'].astype('string').str.strip().str.upper()
    event_mapped = event_text.map({
        'TRUE': 1, 'T': 1, 'Y': 1, 'YES': 1,
        'FALSE': 0, 'F': 0, 'N': 0, 'NO': 0,
    })
    data['ufi_event'] = event_numeric.fillna(event_mapped).astype('Int64')

# 5단계의 ufi_event만 불량률 계산에 사용합니다.
# 기본 5단계 정의: chip/item 내 UFI value가 하나라도 UFI_EVENT_THRESHOLD(기본 0)를
# 초과하면 1, 아니면 0입니다.
data = data[data['ufi_event'].isin([0, 1])].copy()

data['_wafer_x_mm'] = data['_chip_center_global_x'] * POSITION_SCALE_TO_MM
data['_wafer_y_mm'] = data['_chip_center_global_y'] * POSITION_SCALE_TO_MM
data['_chip_pitch_x_mm'] = data['_chip_pitch_x'].abs() * POSITION_SCALE_TO_MM
data['_chip_pitch_y_mm'] = data['_chip_pitch_y'].abs() * POSITION_SCALE_TO_MM
data['_shot_center_x_mm'] = data['shot_center_fcp_x'] * POSITION_SCALE_TO_MM
data['_shot_center_y_mm'] = data['shot_center_fcp_y'] * POSITION_SCALE_TO_MM
data['_wafer_radius_mm'] = np.sqrt(data['_wafer_x_mm'] ** 2 + data['_wafer_y_mm'] ** 2)


def apply_optional_filter(frame, column, values, upper=True):
    normalized = normalize_filter_values(values, upper=upper)
    if not normalized:
        return frame
    compare = frame[column].astype('string').str.strip()
    if upper:
        compare = compare.str.upper()
    return frame[compare.isin(normalized)].copy()


data = apply_optional_filter(data, 'ufi_item_id', TARGET_UFI_IDS)
data = apply_optional_filter(data, 'root_lot_id', TARGET_ROOT_LOTS)
data = apply_optional_filter(data, 'wafer_id', TARGET_WAFERS)
data = apply_optional_filter(data, 'selected_process_id', TARGET_PROCESSES)
data = apply_optional_filter(data, 'selected_step_seq', TARGET_STEPS)
data = apply_optional_filter(data, 'selected_ppid_main_ovl', TARGET_PPIDS)

data = data.dropna(
    subset=[
        *MAP_GROUP_COLUMNS,
        '_wafer_x_mm',
        '_wafer_y_mm',
        '_chip_pitch_x_mm',
        '_chip_pitch_y_mm',
    ]
).copy()

data = data[
    data['_wafer_radius_mm'].le(WAFER_RADIUS_MM * 1.10)
    & data['_chip_pitch_x_mm'].gt(0)
    & data['_chip_pitch_y_mm'].gt(0)
].copy()

if data.empty:
    raise ValueError('필터 적용 후 남은 chip 데이터가 없습니다.')

radius_q99 = data['_wafer_radius_mm'].quantile(0.99)
if radius_q99 > 250:
    raise ValueError(
        f'좌표 반경 99% 값이 {radius_q99:.1f} mm입니다. '
        '좌표가 um라면 POSITION_SCALE_TO_MM = 0.001로 변경하세요.'
    )

print(f'입력 파일: {len(INPUT_PATHS):,}개')
for input_path in INPUT_PATHS:
    print(f'  - {input_path}')
print(f'필터 후 row: {len(data):,}')
print(f'root lot: {data["root_lot_id"].nunique():,}')
print(f'wafer: {data[["root_lot_id", "wafer_id"]].drop_duplicates().shape[0]:,}')
print(f'UFI item: {sorted(data["ufi_item_id"].dropna().unique().tolist())}')


# 같은 주차 파일이 중복 투입되어도 동일 wafer/chip을 두 번 세지 않도록
# 먼저 wafer 단위의 chip 관측치 하나로 정리합니다.
OBSERVATION_KEY_COLUMNS = [
    'root_lot_id',
    'wafer_id',
    *MAP_GROUP_COLUMNS,
    '_wafer_x_mm',
    '_wafer_y_mm',
]

observation_aggregation = {
    '_chip_pitch_x_mm': 'median',
    '_chip_pitch_y_mm': 'median',
    '_shot_center_x_mm': 'median',
    '_shot_center_y_mm': 'median',
    '_source_file': lambda values: ' | '.join(sorted(set(map(str, values.dropna())))),
    'dx_absmax': 'mean',
    'dy_absmax': 'mean',
    'radial_ovl_max': 'mean',
    'ufi_event': 'max',
}

observation_data = (
    data
    .groupby(OBSERVATION_KEY_COLUMNS, as_index=False, dropna=False)
    .agg(observation_aggregation)
)

# 여러 wafer의 같은 물리적 chip 위치를 합칩니다.
# OVL은 평균, UFI는 event chip 수 / 전체 chip 관측 수로 계산합니다.
CHIP_KEY_COLUMNS = [
    *MAP_GROUP_COLUMNS,
    '_wafer_x_mm',
    '_wafer_y_mm',
]

chip_data = (
    observation_data
    .groupby(CHIP_KEY_COLUMNS, as_index=False, dropna=False)
    .agg(
        _chip_pitch_x_mm=('_chip_pitch_x_mm', 'median'),
        _chip_pitch_y_mm=('_chip_pitch_y_mm', 'median'),
        _shot_center_x_mm=('_shot_center_x_mm', 'median'),
        _shot_center_y_mm=('_shot_center_y_mm', 'median'),
        _source_file=(
            '_source_file',
            lambda values: ' | '.join(sorted({
                source_file
                for value in values.dropna().astype(str)
                for source_file in value.split(' | ')
                if source_file
            })),
        ),
        dx_absmax=('dx_absmax', 'mean'),
        dy_absmax=('dy_absmax', 'mean'),
        radial_ovl_max=('radial_ovl_max', 'mean'),
        observation_count=('ufi_event', 'size'),
        ufi_event_count=('ufi_event', 'sum'),
    )
)

chip_data['ufi_rate_pct'] = (
    100.0 * chip_data['ufi_event_count'] / chip_data['observation_count']
)


def calculate_metric_scale(frame, metric):
    config = METRIC_CONFIG[metric]

    if config.get('binary', False):
        return {'kind': 'binary'}

    values = pd.to_numeric(frame[metric], errors='coerce').dropna().to_numpy(float)

    if len(values) == 0:
        return {'kind': 'empty'}

    if metric in MANUAL_COLOR_LIMITS:
        lower, upper = MANUAL_COLOR_LIMITS[metric]
    elif config.get('signed', False):
        limit = float(np.nanpercentile(np.abs(values), ROBUST_COLOR_PERCENTILES[1]))
        limit = max(limit, 1e-9)
        lower, upper = -limit, limit
    else:
        lower = float(np.nanpercentile(values, ROBUST_COLOR_PERCENTILES[0]))
        upper = float(np.nanpercentile(values, ROBUST_COLOR_PERCENTILES[1]))

        if metric in ('dx_absmax', 'dy_absmax', 'radial_ovl_max', 'ufi_rate_pct'):
            lower = min(0.0, lower)

        if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
            lower = float(np.nanmin(values))
            upper = float(np.nanmax(values))

        if lower == upper:
            upper = lower + 1.0

    return {
        'kind': 'signed' if config.get('signed', False) else 'continuous',
        'vmin': float(lower),
        'vmax': float(upper),
    }


METRIC_SCALES = {
    metric: calculate_metric_scale(chip_data, metric)
    for metric in METRIC_CONFIG
}

scale_table = pd.DataFrame([
    {
        'metric': metric,
        **scale,
    }
    for metric, scale in METRIC_SCALES.items()
])

try:
    display(scale_table)
except NameError:
    print(scale_table)


def infer_shot_axis_pitch(center_values, observed_spans):
    centers = pd.to_numeric(center_values, errors='coerce').dropna().to_numpy(float)
    centers = np.unique(np.round(centers, 6))
    center_pitch = np.nan

    if len(centers) >= 2:
        differences = np.diff(np.sort(centers))
        differences = differences[differences > 1e-6]
        if len(differences):
            center_pitch = float(np.min(differences))

    spans = pd.to_numeric(pd.Series(observed_spans), errors='coerce').dropna().to_numpy(float)
    spans = spans[spans > 0]
    observed_pitch = float(np.nanpercentile(spans, 90)) if len(spans) else np.nan

    if np.isfinite(center_pitch) and np.isfinite(observed_pitch):
        # 중간 shot이 없는 경우 center 간격이 실제 pitch의 배수일 수 있다.
        if center_pitch > observed_pitch * 1.5:
            return observed_pitch
        return max(center_pitch, observed_pitch)
    if np.isfinite(center_pitch):
        return center_pitch
    if np.isfinite(observed_pitch):
        return observed_pitch
    return np.nan


def draw_shot_boundaries(axis, wafer_data, wafer_circle):
    if not SHOW_SHOT_BOUNDARIES:
        return

    shot_work = wafer_data.dropna(
        subset=['_shot_center_x_mm', '_shot_center_y_mm']
    ).copy()
    if shot_work.empty:
        return

    shot_work['_shot_center_x_draw'] = shot_work['_shot_center_x_mm'].round(4)
    shot_work['_shot_center_y_draw'] = shot_work['_shot_center_y_mm'].round(4)

    shot_extents = []
    for (center_x, center_y), shot in shot_work.groupby(
        ['_shot_center_x_draw', '_shot_center_y_draw'],
        dropna=False,
    ):
        left = (shot['_wafer_x_mm'] - shot['_chip_pitch_x_mm'] / 2.0).min()
        right = (shot['_wafer_x_mm'] + shot['_chip_pitch_x_mm'] / 2.0).max()
        bottom = (shot['_wafer_y_mm'] - shot['_chip_pitch_y_mm'] / 2.0).min()
        top = (shot['_wafer_y_mm'] + shot['_chip_pitch_y_mm'] / 2.0).max()
        shot_extents.append({
            'center_x': float(center_x),
            'center_y': float(center_y),
            'span_x': float(right - left),
            'span_y': float(top - bottom),
        })

    shot_extents = pd.DataFrame(shot_extents)
    shot_pitch_x = infer_shot_axis_pitch(shot_extents['center_x'], shot_extents['span_x'])
    shot_pitch_y = infer_shot_axis_pitch(shot_extents['center_y'], shot_extents['span_y'])

    if not np.isfinite(shot_pitch_x) or not np.isfinite(shot_pitch_y):
        return

    for shot in shot_extents.itertuples(index=False):
        boundary = Rectangle(
            (shot.center_x - shot_pitch_x / 2.0, shot.center_y - shot_pitch_y / 2.0),
            shot_pitch_x,
            shot_pitch_y,
            fill=False,
            edgecolor=SHOT_BOUNDARY_COLOR,
            linewidth=SHOT_BOUNDARY_LINEWIDTH,
            alpha=SHOT_BOUNDARY_ALPHA,
            zorder=4,
        )
        boundary.set_clip_path(wafer_circle)
        axis.add_patch(boundary)


def draw_metric_map(fig, axis, wafer_data, metric, title=None):
    config = METRIC_CONFIG[metric]
    scale = METRIC_SCALES[metric]
    work = wafer_data.dropna(subset=[metric]).copy()

    if work.empty or scale['kind'] == 'empty':
        axis.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=axis.transAxes)
        axis.set_title(title or config['title'])
        axis.set_axis_off()
        return None

    rectangles = [
        Rectangle(
            (
                wafer_x - pitch_x / 2.0,
                wafer_y - pitch_y / 2.0,
            ),
            pitch_x,
            pitch_y,
        )
        for wafer_x, wafer_y, pitch_x, pitch_y in zip(
            work['_wafer_x_mm'].to_numpy(float),
            work['_wafer_y_mm'].to_numpy(float),
            work['_chip_pitch_x_mm'].to_numpy(float),
            work['_chip_pitch_y_mm'].to_numpy(float),
        )
    ]

    values = work[metric].to_numpy(float)

    if scale['kind'] == 'binary':
        cmap = ListedColormap(['#ECEFF1', '#D32F2F'])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    elif scale['kind'] == 'signed':
        cmap = plt.get_cmap(config['cmap'])
        norm = TwoSlopeNorm(vmin=scale['vmin'], vcenter=0.0, vmax=scale['vmax'])
    else:
        cmap = plt.get_cmap(config['cmap'])
        norm = Normalize(vmin=scale['vmin'], vmax=scale['vmax'], clip=True)

    collection = PatchCollection(
        rectangles,
        cmap=cmap,
        norm=norm,
        edgecolor='#FFFFFF',
        linewidth=0.15,
    )
    collection.set_array(values)
    axis.add_collection(collection)

    wafer_circle = Circle(
        (0, 0),
        WAFER_RADIUS_MM,
        fill=False,
        edgecolor='black',
        linewidth=1.2,
        zorder=5,
    )
    axis.add_patch(wafer_circle)
    draw_shot_boundaries(axis, work, wafer_circle)
    axis.set_xlim(-WAFER_RADIUS_MM - 5, WAFER_RADIUS_MM + 5)
    axis.set_ylim(-WAFER_RADIUS_MM - 5, WAFER_RADIUS_MM + 5)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlabel('Wafer X (mm)')
    axis.set_ylabel('Wafer Y (mm)')
    axis.set_title(title or config['title'], fontsize=11)

    colorbar = fig.colorbar(collection, ax=axis, fraction=0.046, pad=0.04)

    if scale['kind'] == 'binary':
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(['Normal', 'UFI event'])
    else:
        colorbar.set_label(config['unit'])

    return collection


def group_title(group_values, observation_count=None, wafer_count=None):
    suffix = ''
    if observation_count is not None and wafer_count is not None:
        suffix = f' | observations={observation_count:,}, wafers={wafer_count:,}'

    return (
        f'{group_values["selected_process_id"]} | '
        f'{group_values["selected_step_seq"]} | '
        f'{group_values["selected_ppid_main_ovl"]} | '
        f'{group_values["ufi_item_id"]}'
        f'{suffix}'
    )


timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
RUN_OUTPUT_DIR = OUTPUT_ROOT_DIR / f'WAFER_HEATMAP__{timestamp}'
COMPARISON_DIR = RUN_OUTPUT_DIR / 'COMPARISON_4PANEL'
SINGLE_DIR = RUN_OUTPUT_DIR / 'SINGLE_MAPS'

COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
SINGLE_DIR.mkdir(parents=True, exist_ok=True)

map_groups = list(
    chip_data.groupby(MAP_GROUP_COLUMNS, dropna=False, sort=True)
)

index_records = []
first_comparison_path = None

for group_no, (group_key, wafer_data) in enumerate(map_groups, start=1):
    if not isinstance(group_key, tuple):
        group_key = (group_key,)

    group_values = dict(zip(MAP_GROUP_COLUMNS, group_key))
    token = '__'.join(safe_token(group_values[column]) for column in MAP_GROUP_COLUMNS)

    observation_mask = pd.Series(True, index=observation_data.index)
    for column, value in group_values.items():
        observation_mask &= observation_data[column].eq(value)
    group_observations = observation_data.loc[observation_mask]
    group_observation_count = len(group_observations)
    group_wafer_count = group_observations[['root_lot_id', 'wafer_id']].drop_duplicates().shape[0]
    group_root_lot_count = group_observations['root_lot_id'].nunique()
    title = group_title(group_values, group_observation_count, group_wafer_count)

    comparison_path = ''

    if SAVE_COMPARISON_PANEL:
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 12.0))

        for axis, metric in zip(axes.flat, COMPARISON_METRICS):
            draw_metric_map(fig, axis, wafer_data, metric)

        fig.suptitle(title, fontsize=13, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        comparison_file = COMPARISON_DIR / f'COMPARE__{token}.png'
        fig.savefig(comparison_file, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        comparison_path = str(comparison_file)

        if first_comparison_path is None:
            first_comparison_path = comparison_file

    single_paths = []

    if SAVE_SINGLE_MAPS:
        group_single_dir = SINGLE_DIR / safe_token(group_values['ufi_item_id'])
        group_single_dir.mkdir(parents=True, exist_ok=True)

        for metric in SINGLE_MAP_METRICS:
            fig, axis = plt.subplots(figsize=(7.2, 6.6))
            draw_metric_map(fig, axis, wafer_data, metric, title=METRIC_CONFIG[metric]['title'])
            fig.suptitle(title, fontsize=10.5, y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.965))
            single_file = group_single_dir / f'{safe_token(metric)}__{token}.png'
            fig.savefig(single_file, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close(fig)
            single_paths.append(str(single_file))

    record = {
        **group_values,
        'chip_position_count': len(wafer_data),
        'root_lot_count': group_root_lot_count,
        'wafer_count': group_wafer_count,
        'observation_count': int(wafer_data['observation_count'].sum()),
        'ufi_event_count': int(wafer_data['ufi_event_count'].sum()),
        'overall_ufi_rate_pct': (
            100.0 * wafer_data['ufi_event_count'].sum()
            / wafer_data['observation_count'].sum()
        ),
        'source_files': ' | '.join(sorted({
            source_file
            for value in wafer_data['_source_file'].dropna().astype(str)
            for source_file in value.split(' | ')
            if source_file
        })),
        'comparison_path': comparison_path,
        'single_map_count': len(single_paths),
        'single_map_paths': ' | '.join(single_paths),
    }
    index_records.append(record)

map_index = pd.DataFrame(index_records)
index_path = RUN_OUTPUT_DIR / 'WAFER_MAP_INDEX.xlsx'

with pd.ExcelWriter(index_path, engine='openpyxl') as writer:
    map_index.to_excel(writer, sheet_name='MAP_INDEX', index=False)
    scale_table.to_excel(writer, sheet_name='COLOR_SCALE', index=False)

print(f'map group: {len(map_groups):,}')
print(f'결과 폴더: {RUN_OUTPUT_DIR}')
print(f'index Excel: {index_path}')

if PREVIEW_FIRST_COMPARISON and first_comparison_path is not None:
    try:
        from IPython.display import Image, display
        display(Image(filename=str(first_comparison_path)))
    except Exception:
        pass


