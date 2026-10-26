"""Rebuild readable tables and optional figures from measured run artifacts."""
import argparse
import csv
import json
from pathlib import Path


def read(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def write_csv(path, rows, default_fields):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or default_fields
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def export(run_dir, plots=False):
    run_dir = Path(run_dir)
    config = read(run_dir/'comparison_config.json', {})
    if not config:
        raise ValueError('comparison_config.json is required')
    tasks = config['settings']['config']['task_sequence']
    epochs = read(run_dir/'epoch_metrics.json', [])
    flat = []
    for row in epochs:
        flat.append({**{k:v for k,v in row.items() if k != 'validation'},
                     **{'val_'+k:v for k,v in row.get('validation', {}).items()}})
    write_csv(run_dir/'epoch_metrics.csv',flat,['stage','task','epoch','loss'])
    matrix = read(run_dir/'accuracy_matrix.json', [])
    matrix_rows = [{**{k:v for k,v in row.items() if k != 'accuracies'},
                    **{task:(row.get('accuracies') or {}).get(task) for task in tasks}} for row in matrix]
    write_csv(run_dir/'accuracy_matrix.csv',matrix_rows,['stage','task',*tasks])
    result = read(run_dir/'final_test.json', {})
    write_csv(run_dir/'final_test.csv',[{'condition':k,'accuracy_percent':v} for k,v in result.items()],
              ['condition','accuracy_percent'])
    references = read(run_dir/'replay_references.json', {})
    missing = references.get('missing_tasks', tasks)
    notes = [
        '# 运行结果说明', '',
        '本目录的 CSV 由已有检查点、保存的训练记录和实际重新评估的结果生成。', '',
        f"- 消融设置：{config['settings']['ablation']['name']}",
        f"- 训练随机种子：{config['settings']['config']['seed']}",
        f"- 导入历史检查点：{bool(config.get('imported_history'))}",
        f'- 已有逐轮训练记录：{len(epochs)} 条',
        f'- 已完成测试项目：{", ".join(result) or "暂无"}',
        f'- 缺少首次学习基准的任务：{", ".join(missing) or "无"}', '',
        'JSON 中 null 和 CSV 中空白表示原文件未保存或尚未测量。损失、耗时和重放轨迹只包含已有记录；'
        '任务边界准确率可由对应模型重新评估。旧版导入及缺失状态详见 checkpoint_imports.json、'
        'resource_metrics.json 和 stage*_complete.json。', '',
        '绘图命令：python export_run_artifacts.py --run-dir "本运行目录" --plots',
    ]
    (run_dir/'RESULTS_CN.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    if plots:
        import os
        os.environ.setdefault('MPLCONFIGDIR', str((run_dir/'.plot_cache').resolve()))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        if epochs:
            fig, ax = plt.subplots(figsize=(8,4))
            for task in tasks:
                local = [r for r in epochs if r['task'] == task]
                if local:
                    ax.plot([r['epoch']+1 for r in local],[r['loss'] for r in local],label=task)
            ax.set(xlabel='Epoch within task',ylabel='Training loss'); ax.legend()
            fig.tight_layout(); fig.savefig(run_dir/'training_curves.png',dpi=180); plt.close(fig)
        if matrix:
            data = np.array([[(r.get('accuracies') or {}).get(t,float('nan')) for t in tasks] for r in matrix])
            fig, ax = plt.subplots(figsize=(6,4))
            shown = ax.imshow(np.ma.masked_invalid(data),vmin=0,vmax=100,cmap='Blues')
            ax.set_xticks(range(len(tasks)),tasks); ax.set_yticks(range(len(matrix)),[r['task'] for r in matrix])
            ax.set(xlabel='Validation task',ylabel='Selected task boundary')
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if np.isfinite(data[i,j]):
                        ax.text(j,i,f'{data[i,j]:.1f}',ha='center',va='center',color='white' if data[i,j]>55 else 'black')
            fig.colorbar(shown,ax=ax,label='Accuracy (%)'); fig.tight_layout()
            fig.savefig(run_dir/'validation_matrix.png',dpi=180); plt.close(fig)
        if result:
            fig,ax = plt.subplots(figsize=(9,4))
            ax.bar(result.keys(),result.values()); ax.set(ylabel='Test accuracy (%)',ylim=(0,100))
            ax.tick_params(axis='x',rotation=25); fig.tight_layout()
            fig.savefig(run_dir/'final_accuracy.png',dpi=180); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--plots',action='store_true')
    args = p.parse_args(); export(args.run_dir,args.plots)


if __name__ == '__main__':
    main()
