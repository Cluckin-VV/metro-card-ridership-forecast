"""Generate publication-sized Chinese figures, tables, animation and self-contained dashboard."""
import base64
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from data_pipeline import ROOT, load_clean

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'figure.dpi':120,'savefig.dpi':180})
FIG=ROOT/'results/figures';FIG.mkdir(exist_ok=True)
BLUE='#145e83';ORANGE='#cb6a2f';TEAL='#247f70'


def save(name):
    plt.tight_layout();plt.savefig(FIG/name,bbox_inches='tight');plt.close()


def main():
    data,audit=load_clean();dev=data.loc[:'2026-01-31']
    p=pd.read_csv(ROOT/'results/predictions.csv',parse_dates=['date']);test=p[p.split=='test'].copy()
    m=json.loads((ROOT/'results/metrics.json').read_text());f=json.loads((ROOT/'results/next_forecast.json').read_text())
    # Do not connect lines across unobserved calendar gaps without showing the gaps.
    calendar=data.card_total.reindex(pd.date_range(data.index.min(),data.index.max()))
    fig,ax=plt.subplots(figsize=(10,3.6));ax.plot(calendar.index,calendar/1e4,color=BLUE,lw=.9)
    ax.scatter(data.index,data.card_total/1e4,s=6,color=BLUE,alpha=.6)
    ax.axvline(pd.Timestamp('2026-02-01'),color=ORANGE,ls='--',label='开发期结束')
    ax.axvline(pd.Timestamp('2026-05-01'),color=TEAL,ls='--',label='测试期开始')
    ax.set(title='图1 卡介质客流与时间切分',ylabel='客流（万人次/日）',xlabel='日期');ax.legend(loc='lower left');save('01_history.png')

    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    axs[0].hist(dev.card_total/1e4,bins=25,color=BLUE,alpha=.85);axs[0].set(title='图2a 开发期客流分布',xlabel='万人次/日',ylabel='观测日数')
    vals=[dev.loc[dev.index.dayofweek==i,'card_total']/1e4 for i in range(7)]
    axs[1].boxplot(vals,tick_labels=['一','二','三','四','五','六','日'],showfliers=True)
    axs[1].set(title='图2b 开发期星期效应',xlabel='星期',ylabel='万人次/日');save('02_eda.png')

    cols=['Total Smart Cards','Total NCMC','Total Tokens','Group Ticket','card_total']
    corr=dev[cols].corr();fig,ax=plt.subplots(figsize=(6.5,4))
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');names=['智能卡','NCMC','单程票','团体票','卡介质合计']
    ax.set_xticks(range(5),names,rotation=20);ax.set_yticks(range(5),names)
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',color='white' if abs(corr.iloc[i,j])>.6 else 'black')
    fig.colorbar(im,ax=ax);ax.set_title('图3 开发期相关系数 仅用于描述');save('03_correlation.png')

    fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    counts=data.resample('MS').size();days=pd.Series(1,index=calendar.index).resample('MS').size()
    axs[0].bar(counts.index,counts,width=20,color=BLUE,label='有记录');axs[0].bar(counts.index,days-counts,bottom=counts,width=20,color='#d7e1e6',label='缺失')
    axs[0].set(title='图4a 月度采集完整性',ylabel='天数');axs[0].tick_params(axis='x',rotation=30);axs[0].legend()
    axs[1].scatter(data.index,data.qr_gap,s=9,color=ORANGE);axs[1].axhline(0,color='gray',lw=.8)
    axs[1].set(title='图4b QR子项和减去汇总',ylabel='人次差额');axs[1].tick_params(axis='x',rotation=30);save('04_data_quality.png')

    cv=pd.read_csv(ROOT/'results/regression_cv.csv');ranking=cv.groupby('model').MAE.agg(['mean','std']).sort_values('mean')
    fig,ax=plt.subplots(figsize=(9,4));ax.barh(ranking.index,ranking['mean']/1e4,xerr=ranking['std']/1e4,color=BLUE)
    ax.invert_yaxis();ax.set(title='图5 开发期三折时间顺序验证',xlabel='平均MAE ± 折间标准差（万人次/日）');save('05_model_comparison.png')

    fig,ax=plt.subplots(figsize=(10,3.8));ax.fill_between(test.date,test.lower90/1e4,test.upper90/1e4,color=BLUE,alpha=.12,label='名义90%区间')
    ax.plot(test.date,test.actual/1e4,'o-',ms=2.6,lw=1,color=BLUE,label='观测值')
    ax.plot(test.date,test.predicted/1e4,ls='--',lw=1,color=ORANGE,label='预测值')
    ax.set(title='图6 测试期滚动预测 缺失日不评估',ylabel='客流（万人次/日）',xlabel='日期');ax.legend(ncol=3,fontsize=9);save('06_predictions.png')

    fig,axs=plt.subplots(1,2,figsize=(9,3.7));cm=np.array(m['test']['classification']['confusion_matrix'])
    axs[0].imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):axs[0].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>20 else 'black',fontsize=16)
    axs[0].set(xticks=[0,1],yticks=[0,1],xticklabels=['常规','高需求'],yticklabels=['常规','高需求'],xlabel='预测类别',ylabel='真实类别',title='图7a 逻辑回归混淆矩阵')
    err=test.actual-test.predicted;axs[1].scatter(test.predicted/1e4,err/1e4,c=np.where(test.actual_high==1,BLUE,ORANGE),s=20)
    axs[1].axhline(0,color='gray',ls='--');axs[1].set(title='图7b 预测残差',xlabel='预测（万人次/日）',ylabel='实际减预测（万人次/日）');save('07_classification_errors.png')

    cl=pd.read_csv(ROOT/'results/clusters.csv',parse_dates=['date']);fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    colors=[TEAL,BLUE,ORANGE]
    for label,g in cl.groupby('cluster'):
        axs[0].scatter(g['Total NCMC']/g.card_total,g.card_total/1e4,s=11,alpha=.7,label=f'日型{label}',color=colors[label%3])
        axs[1].scatter(g.date,g.card_total/1e4,s=10,alpha=.7,label=f'日型{label}',color=colors[label%3])
    axs[0].set(title='图8a 票种结构与日型',xlabel='NCMC占卡介质比例',ylabel='万人次/日');axs[0].legend()
    axs[1].set(title='图8b 日型的时间分布',ylabel='万人次/日');axs[1].tick_params(axis='x',rotation=25);save('08_clusters.png')

    data.describe().to_csv(ROOT/'results/descriptive_statistics.csv',encoding='utf-8-sig')
    dev.groupby(dev.index.dayofweek).card_total.agg(['count','mean','median','std']).to_csv(ROOT/'results/weekday_statistics.csv')
    # Short replay animation is an optional teaching artifact, not evidence of live data.
    fig,ax=plt.subplots(figsize=(8,3.5));ax.set_xlim(test.date.min(),test.date.max());ax.set_ylim(0,max(test.upper90)/1e4*1.05)
    actual_line,=ax.plot([],[],color=BLUE,label='实际');pred_line,=ax.plot([],[],color=ORANGE,label='预测')
    ax.set(title='测试期历史回放',ylabel='万人次/日');ax.legend();fig.autofmt_xdate()
    def update(n):
        g=test.iloc[:n];actual_line.set_data(g.date,g.actual/1e4);pred_line.set_data(g.date,g.predicted/1e4);return actual_line,pred_line
    FuncAnimation(fig,update,frames=list(range(1,len(test)+1,3))+[len(test)],interval=180).save(FIG/'prediction_replay.gif',writer=PillowWriter(fps=5));plt.close()
    payload={'metrics':m,'forecast':f,'audit':audit,'predictions':json.loads(test.to_json(orient='records',date_format='iso'))}
    (ROOT/'results/dashboard_data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print('8 figures, descriptive tables, GIF and dashboard payload generated')


if __name__=='__main__':main()
