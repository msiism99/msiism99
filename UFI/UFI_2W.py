# 6-2단계 2WEEKS

"""
주차별 UFI_BOWL_CURVES PKL을 이용해 연속 2주 rolling UFI curve를 다시 그린다.

입력 예:
2WEEK_INPUT/
    2632__UFI_BOWL_CURVES.pkl
    2633__UFI_BOWL_CURVES.pkl
    ...

출력 window:
2632 + 2633 -> 2633
2633 + 2634 -> 2634
...

Stage 1~5, RBF fitting은 다시 실행하지 않는다.
"""

from pathlib import Path
from datetime import datetime
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# 1. 사용자 설정
# =========================================================
INPUT_DIR = Path.cwd() / '2WEEK_INPUT'
OUTPUT_ROOT_DIR = Path.cwd() / '2WEEK_OUTPUT'

# 그릴 UFI item. 원하는 번호를 추가/삭제하면 된다.
TARGET_UFI_IDS = ['UFI1019', 'UFI1027']

# 두 주를 합친 다음 적용한다.
# 예: 1주차 n=12, 2주차 n=15이면 통합 n=27이므로 표시된다.
MIN_BIN_CHIP_COUNT_FOR_PLOT = 20

SHOW_CONFIDENCE_BAND = True
SHOW_BIN_COUNT_LABELS = False

# None: 모든 rolling window와 모든 그래프에 동일한 y축을 자동 적용.
# 숫자 입력: 해당 값으로 고정. 예: COMMON_Y_MAX = 30
COMMON_Y_MAX = None
COMMON_Y_TICK = 5

FIGURE_SIZE = (10.5, 6.5)
FIGURE_DPI = 180

CURVE_GROUP_COLUMNS = [
    'selected_process_id',
    'selected_step_seq',
    'selected_ppid_main_ovl',
    'ufi_item_id',
]

BIN_KEY_COLUMNS = [
    *CURVE_GROUP_COLUMNS,
    'curve_axis',
    'shot_scope',
    'direction',
    'source_metric',
    'ovl_bin_left',
    'ovl_bin_right',
    'ovl_bin_center',
    'bin_width_nm',
]

STRING_COLUMNS = [
    *CURVE_GROUP_COLUMNS,
    'curve_axis',
    'shot_scope',
    'direction',
    'source_metric',
]

NUMERIC_COLUMNS = [
    'ovl_bin_left',
    'ovl_bin_right',
    'ovl_bin_center',
    'bin_width_nm',
    'chip_count',
    'ufi_event_count',
    'ovl_value_mean',
    'ovl_value_min',
    'ovl_value_max',
]

COLOR_MAP = {
    'NEGATIVE': '#1565C0',
    'POSITIVE': '#C62828',
    'RADIAL': '#2E7D32',
}

FILL_COLOR_MAP = {
    'NEGATIVE': '#64B5F6',
    'POSITIVE': '#EF9A9A',
    'RADIAL': '#81C784',
}


# =========================================================
# 2. 공통 함수
# =========================================================
def safe_token(value):
    token = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value).strip())
    return token.strip('._') or 'NA'


def normalize_value(value):
    return str(value).strip().upper()


def parse_week_tag(path):
    match = re.match(r'^(\d{4})__UFI_BOWL_CURVES(?:__.*)?\.pkl$', path.name, re.I)
    if not match:
        return None

    tag = match.group(1)
    year = int(tag[:2])
    week = int(tag[2:])

    if not 1 <= week <= 53:
        raise ValueError(f'올바르지 않은 주차 tag입니다: {path.name}')

    return tag


def is_consecutive_week(previous_tag, current_tag):
    previous_year, previous_week = int(previous_tag[:2]), int(previous_tag[2:])
    current_year, current_week = int(current_tag[:2]), int(current_tag[2:])

    if current_year == previous_year and current_week == previous_week + 1:
        return True

    return (
        current_year == previous_year + 1
        and current_week == 1
        and previous_week in (52, 53)
    )


def binomial_rate_ci(event_count, total_count, z=1.96):
    """Wilson 95% CI. 반환 단위는 percent."""
    if total_count <= 0:
        return np.nan, np.nan

    p = event_count / total_count
    denominator = 1.0 + z ** 2 / total_count
    center = (p + z ** 2 / (2.0 * total_count)) / denominator
    half_width = (
        z
        * math.sqrt(
            p * (1.0 - p) / total_count
            + z ** 2 / (4.0 * total_count ** 2)
        )
        / denominator
    )

    return (
        max(0.0, center - half_width) * 100.0,
        min(1.0, center + half_width) * 100.0,
    )


def validate_and_prepare_weekly_data(data, path, target_ufi_ids):
    required = set(BIN_KEY_COLUMNS + ['chip_count', 'ufi_event_count'])
    missing = required - set(data.columns)

    if missing:
        raise KeyError(
            f'{path.name}에 필수 컬럼이 없습니다: {sorted(missing)}'
        )

    result = data.copy()

    for column in STRING_COLUMNS:
        result[column] = result[column].astype('string').str.strip().str.upper()

    for column in NUMERIC_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors='coerce')

    result = result[
        result['ufi_item_id'].isin(target_ufi_ids)
    ].copy()

    result = result.dropna(
        subset=BIN_KEY_COLUMNS + ['chip_count', 'ufi_event_count']
    )

    result = result[
        result['chip_count'].gt(0)
        & result['ufi_event_count'].ge(0)
        & result['ufi_event_count'].le(result['chip_count'])
    ].copy()

    return result


def combine_two_weeks(previous_data, current_data, previous_tag, current_tag):
    combined = pd.concat([previous_data, current_data], ignore_index=True)

    if combined.empty:
        return pd.DataFrame()

    if 'ovl_value_mean' in combined.columns:
        combined['_ovl_weighted_sum'] = (
            combined['ovl_value_mean'] * combined['chip_count']
        )
    else:
        combined['_ovl_weighted_sum'] = np.nan

    aggregation = {
        'chip_count': ('chip_count', 'sum'),
        'ufi_event_count': ('ufi_event_count', 'sum'),
        '_ovl_weighted_sum': ('_ovl_weighted_sum', 'sum'),
    }

    if 'ovl_value_min' in combined.columns:
        aggregation['ovl_value_min'] = ('ovl_value_min', 'min')

    if 'ovl_value_max' in combined.columns:
        aggregation['ovl_value_max'] = ('ovl_value_max', 'max')

    result = (
        combined
        .groupby(BIN_KEY_COLUMNS, as_index=False, dropna=False, sort=False)
        .agg(**aggregation)
    )

    result['ovl_value_mean'] = np.divide(
        result['_ovl_weighted_sum'],
        result['chip_count'],
        out=np.full(len(result), np.nan, dtype=float),
        where=result['chip_count'].to_numpy(float) > 0,
    )
    result = result.drop(columns='_ovl_weighted_sum')

    result['ufi_rate_pct'] = (
        result['ufi_event_count'] / result['chip_count'] * 100.0
    )

    confidence = result.apply(
        lambda row: binomial_rate_ci(
            int(row['ufi_event_count']),
            int(row['chip_count']),
        ),
        axis=1,
        result_type='expand',
    )
    confidence.columns = ['ufi_rate_ci_low_pct', 'ufi_rate_ci_high_pct']
    result = pd.concat([result, confidence], axis=1)

    result.insert(0, 'output_week', current_tag)
    result.insert(0, 'source_week_to', current_tag)
    result.insert(0, 'source_week_from', previous_tag)

    return result


# =========================================================
# 3. 주차별 파일 확인
# =========================================================
if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f'입력 폴더가 없습니다: {INPUT_DIR}\n'
        '2WEEK_INPUT 폴더를 만들고 주차별 PKL을 넣어주세요.'
    )

target_ufi_ids = sorted({
    normalize_value(item_id)
    for item_id in TARGET_UFI_IDS
    if str(item_id).strip()
})

if not target_ufi_ids:
    raise ValueError('TARGET_UFI_IDS가 비어 있습니다.')

weekly_files = {}
ignored_files = []

for path in sorted(INPUT_DIR.glob('*.pkl')):
    week_tag = parse_week_tag(path)

    if week_tag is None:
        ignored_files.append(path.name)
        continue

    if week_tag in weekly_files:
        raise ValueError(
            f'{week_tag} 주차 파일이 2개 이상입니다. 주차별로 하나만 남겨주세요.\n'
            f'- {weekly_files[week_tag]}\n- {path}'
        )

    weekly_files[week_tag] = path

if len(weekly_files) < 2:
    raise FileNotFoundError(
        '연속된 2주 그래프를 만들려면 주차별 PKL이 최소 2개 필요합니다.\n'
        f'검색 위치: {INPUT_DIR}'
    )

sorted_week_tags = sorted(
    weekly_files,
    key=lambda tag: (int(tag[:2]), int(tag[2:])),
)

window_pairs = []
missing_pair_messages = []

for previous_tag, current_tag in zip(sorted_week_tags[:-1], sorted_week_tags[1:]):
    if is_consecutive_week(previous_tag, current_tag):
        window_pairs.append((previous_tag, current_tag))
    else:
        missing_pair_messages.append(
            f'{previous_tag}와 {current_tag}는 연속 주차가 아니므로 결합하지 않음'
        )

if not window_pairs:
    raise ValueError(
        '연속된 주차 파일 조합이 없습니다.\n'
        + '\n'.join(missing_pair_messages)
    )


# =========================================================
# 4. 주차별 파일은 한 번만 읽고 대상 UFI만 보관
# =========================================================
weekly_data = {}

for week_tag in sorted_week_tags:
    path = weekly_files[week_tag]
    data = pd.read_pickle(path)
    weekly_data[week_tag] = validate_and_prepare_weekly_data(
        data,
        path,
        target_ufi_ids,
    )

found_ufi_ids = sorted({
    item_id
    for data in weekly_data.values()
    for item_id in data['ufi_item_id'].dropna().astype(str).unique()
})

missing_target_ufi_ids = sorted(set(target_ufi_ids) - set(found_ufi_ids))

if not found_ufi_ids:
    raise ValueError(
        f'입력 파일에서 대상 UFI를 찾지 못했습니다: {target_ufi_ids}'
    )


# =========================================================
# 5. 연속 2주 rolling bin 결과 생성
# =========================================================
window_results = {}

for previous_tag, current_tag in window_pairs:
    combined = combine_two_weeks(
        weekly_data[previous_tag],
        weekly_data[current_tag],
        previous_tag,
        current_tag,
    )

    if not combined.empty:
        window_results[current_tag] = combined

if not window_results:
    raise ValueError('대상 UFI에 대해 생성된 2주 통합 결과가 없습니다.')


# =========================================================
# 6. 전체 rolling 결과 공통 y축 결정
# =========================================================
plot_candidates = []

for combined in window_results.values():
    valid = combined[
        combined['chip_count'] >= MIN_BIN_CHIP_COUNT_FOR_PLOT
    ]

    if not valid.empty:
        plot_candidates.append(valid)

if not plot_candidates:
    raise ValueError(
        '두 주를 합친 뒤에도 표시 가능한 bin이 없습니다.\n'
        f'MIN_BIN_CHIP_COUNT_FOR_PLOT={MIN_BIN_CHIP_COUNT_FOR_PLOT}'
    )

all_plot_data = pd.concat(plot_candidates, ignore_index=True)

if COMMON_Y_MAX is None:
    y_source_column = (
        'ufi_rate_ci_high_pct'
        if SHOW_CONFIDENCE_BAND
        else 'ufi_rate_pct'
    )
    max_y = all_plot_data[y_source_column].max()
    common_y_max = max(COMMON_Y_TICK, float(max_y) * 1.05)
    common_y_max = math.ceil(common_y_max / COMMON_Y_TICK) * COMMON_Y_TICK
    common_y_max = min(100.0, common_y_max)
else:
    common_y_max = float(COMMON_Y_MAX)

if not np.isfinite(common_y_max) or common_y_max <= 0:
    raise ValueError(f'COMMON_Y_MAX가 올바르지 않습니다: {common_y_max}')


# =========================================================
# 7. 그래프 함수
# =========================================================
def plot_axis_curve(axis_curve_data, group_values, previous_tag, current_tag, output_path):
    curve_axis = str(group_values['curve_axis'])
    shot_scope = str(group_values['shot_scope'])
    fig, axis = plt.subplots(figsize=FIGURE_SIZE)
    plotted = False
    all_x = []

    for direction in ['NEGATIVE', 'POSITIVE', 'RADIAL']:
        plot_data = axis_curve_data[
            axis_curve_data['direction'].eq(direction)
            & axis_curve_data['chip_count'].ge(MIN_BIN_CHIP_COUNT_FOR_PLOT)
        ].sort_values('ovl_bin_center')

        if plot_data.empty:
            continue

        plotted = True
        x = plot_data['ovl_bin_center'].to_numpy(float)
        y = plot_data['ufi_rate_pct'].to_numpy(float)
        ci_low = plot_data['ufi_rate_ci_low_pct'].to_numpy(float)
        ci_high = plot_data['ufi_rate_ci_high_pct'].to_numpy(float)
        all_x.extend(x[np.isfinite(x)].tolist())

        axis.plot(
            x,
            y,
            marker='o',
            markersize=5,
            linewidth=1.8,
            color=COLOR_MAP[direction],
            label=direction.title(),
        )

        if SHOW_CONFIDENCE_BAND:
            axis.fill_between(
                x,
                ci_low,
                ci_high,
                color=FILL_COLOR_MAP[direction],
                alpha=0.22,
                linewidth=0,
            )

        if SHOW_BIN_COUNT_LABELS:
            for row in plot_data.itertuples(index=False):
                axis.annotate(
                    f'n={int(row.chip_count)}',
                    (row.ovl_bin_center, row.ufi_rate_pct),
                    xytext=(0, 6),
                    textcoords='offset points',
                    ha='center',
                    fontsize=7,
                    alpha=0.72,
                )

    if not plotted:
        plt.close(fig)
        return False

    max_abs_x = max([abs(value) for value in all_x] or [0.1])
    x_padding = max(0.1, max_abs_x * 0.05)

    if curve_axis == 'RADIAL':
        axis.set_xlim(0, max_abs_x + x_padding)
    else:
        axis.set_xlim(-(max_abs_x + x_padding), max_abs_x + x_padding)

    axis.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    axis.set_ylim(0, common_y_max)
    axis.set_xlabel(f'{curve_axis} OVL (0.1 nm bin center)')
    axis.set_ylabel('UFI rate (%)')
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    title_values = [
        group_values['selected_process_id'],
        group_values['selected_step_seq'],
        group_values['selected_ppid_main_ovl'],
        group_values['ufi_item_id'],
    ]
    axis.set_title(
        f'{curve_axis} UFI curve — {shot_scope}\n'
        f'{previous_tag} + {current_tag} → {current_tag} / '
        + ' / '.join(map(str, title_values)),
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    return True


# =========================================================
# 8. 주차별 결과 저장
# =========================================================
run_timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
RUN_OUTPUT_DIR = OUTPUT_ROOT_DIR / f'ROLLING_2WEEK__{run_timestamp}'
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

window_summary_records = []
all_window_frames = []

chart_group_columns = CURVE_GROUP_COLUMNS + ['curve_axis', 'shot_scope']

for previous_tag, current_tag in window_pairs:
    if current_tag not in window_results:
        continue

    combined = window_results[current_tag]
    week_output_dir = RUN_OUTPUT_DIR / current_tag
    chart_root_dir = week_output_dir / 'CHARTS'
    chart_root_dir.mkdir(parents=True, exist_ok=True)

    chart_count = 0

    for group_key, chart_data in combined.groupby(
        chart_group_columns,
        dropna=False,
        sort=True,
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_values = dict(zip(chart_group_columns, group_key))
        shot_scope = str(group_values['shot_scope'])
        curve_axis = str(group_values['curve_axis'])
        scope_dir = chart_root_dir / safe_token(shot_scope)
        scope_dir.mkdir(parents=True, exist_ok=True)

        filename = 'UFI_BOWL__' + '__'.join([
            safe_token(group_values['selected_process_id']),
            safe_token(group_values['selected_step_seq']),
            safe_token(group_values['selected_ppid_main_ovl']),
            safe_token(group_values['ufi_item_id']),
            safe_token(curve_axis),
        ]) + '.png'

        saved = plot_axis_curve(
            chart_data,
            group_values,
            previous_tag,
            current_tag,
            scope_dir / filename,
        )
        chart_count += int(saved)

    pickle_path = week_output_dir / f'TWO_WEEK_UFI_BOWL_CURVES__{current_tag}.pkl'
    excel_path = week_output_dir / f'TWO_WEEK_UFI_BOWL_CURVES__{current_tag}.xlsx'
    combined.to_pickle(pickle_path)

    info = pd.DataFrame([{
        'source_week_from': previous_tag,
        'source_week_to': current_tag,
        'output_week': current_tag,
        'target_ufi_ids': ', '.join(target_ufi_ids),
        'minimum_bin_chip_count': MIN_BIN_CHIP_COUNT_FOR_PLOT,
        'common_y_max_pct': common_y_max,
        'combined_bin_rows': len(combined),
        'saved_charts': chart_count,
    }])

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        combined.to_excel(writer, sheet_name='BINNED_UFI_RATE', index=False)
        info.to_excel(writer, sheet_name='WINDOW_INFO', index=False)

    all_window_frames.append(combined)
    window_summary_records.append(info.iloc[0].to_dict())

all_windows = pd.concat(all_window_frames, ignore_index=True)
window_summary = pd.DataFrame(window_summary_records)

all_windows.to_pickle(RUN_OUTPUT_DIR / 'ALL_ROLLING_2WEEK_BINS.pkl')
with pd.ExcelWriter(RUN_OUTPUT_DIR / 'ROLLING_2WEEK_SUMMARY.xlsx', engine='openpyxl') as writer:
    window_summary.to_excel(writer, sheet_name='WINDOW_SUMMARY', index=False)
    all_windows.to_excel(writer, sheet_name='ALL_BINNED_DATA', index=False)
    pd.DataFrame({'ignored_input_file': ignored_files}).to_excel(
        writer, sheet_name='IGNORED_FILES', index=False
    )
    pd.DataFrame({'message': missing_pair_messages}).to_excel(
        writer, sheet_name='SKIPPED_WEEK_PAIRS', index=False
    )

print(f'대상 UFI: {target_ufi_ids}')
print(f'발견 UFI: {found_ufi_ids}')
if missing_target_ufi_ids:
    print(f'데이터에 없는 대상 UFI: {missing_target_ufi_ids}')
print(f'생성 rolling window: {len(window_summary):,}개')
print(f'공통 y축: 0 ~ {common_y_max:g}%')
print(f'결과 폴더: {RUN_OUTPUT_DIR}')

