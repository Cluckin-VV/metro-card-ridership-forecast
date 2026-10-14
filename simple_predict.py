"""静态CSV客流预测：仅使用Python标准库，运行 python simple_predict.py。"""
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'data/raw/NammaMetro_Ridership_Dataset.csv'
OUT = ROOT / 'simple_results'


def load_data(path):
    data = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            day = datetime.strptime(row['Record Date'], '%d-%m-%Y').date()
            values = [float(row[c]) for c in ('Total Smart Cards', 'Total NCMC')]
            if any(not (0 <= v < float('inf')) or not v.is_integer() for v in values):
                raise ValueError(f'{day}存在无效计数')
            total = sum(values)
            if total <= 0 or day in data:
                raise ValueError(f'{day}客流非正或日期重复')
            data[day] = total
    if len(data) < 30:
        raise ValueError('至少需要30条历史记录')
    return data


def predict(data, day):
    # 假设前一天发出预测、数据晚一天发布，故最多使用day-2的数据。
    cutoff = day - timedelta(days=2)
    history = {d: v for d, v in data.items() if d <= cutoff}
    if not history or (day - max(history)).days > 7:
        raise ValueError('可用历史不足或距预测日超过7天')
    # 按实际日历找过去四周的同星期；缺失日不当作零客流。
    values = [history[d] for lag in (7, 14, 21, 28)
              if (d := day - timedelta(days=lag)) in history]
    if not values:
        values = [v for d, v in history.items()
                  if day - timedelta(days=29) <= d <= cutoff]
    return mean(values) if values else history[max(history)]


def main():
    data = load_data(CSV)
    days = sorted(data)
    # 最后75个有记录日用于历史滚动检验；短数据取末尾20%。
    count = min(75, max(1, len(days) // 5))
    rows = []
    for day in days[-count:]:
        estimate = predict(data, day)
        actual = data[day]
        rows.append([str(day), actual, estimate, abs(actual - estimate)])
    future = days[-1] + timedelta(days=2)
    summary = {'target': '智能卡与NCMC合计人次/日', 'observations': len(days),
               'test_days': count, 'MAE': mean(r[3] for r in rows),
               'MAPE_percent': mean(r[3] / r[1] for r in rows) * 100,
               'forecast_date': str(future), 'prediction': predict(data, future),
               'note': '固定历史数据的演示预测，非当前实时预测'}
    OUT.mkdir(exist_ok=True)
    with (OUT / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'actual', 'prediction', 'absolute_error'])
        writer.writerows(rows)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
