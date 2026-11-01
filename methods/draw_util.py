import csv
import warnings
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def plot_gantt_chart(env_idx: int, op_status: np.ndarray, true_op_ct: np.ndarray, true_op_pt: np.ndarray, op_to_job_map: np.ndarray, op_machine_assignment: np.ndarray, env_num_ops: int, num_machines: int, filename: str, title: str, event_time=0.0, TIME_WINDOW_T_w=0.0, gantt_flag=None):
    """Export the scheduled operations as a Gantt chart."""
    schedule_rows = []
    visible_job_ids = set()
    for op_idx in range(env_num_ops):
        status = op_status[op_idx]
        if status == 1 or status == 2:
            machine_idx = op_machine_assignment[op_idx]
            end_time = true_op_ct[op_idx]
            if machine_idx != -1 and np.isfinite(end_time) and (0 <= machine_idx < num_machines):
                processing_time = true_op_pt[op_idx, machine_idx]
                if processing_time > 0 and np.isfinite(processing_time):
                    start_time = end_time - processing_time
                    job_id = op_to_job_map[op_idx]
                    if gantt_flag is not None:
                        flag = gantt_flag[op_idx]
                    else:
                        flag = ''
                    if job_id != -1:
                        schedule_rows.append({'job_id': job_id, 'op_id': op_idx, 'machine_id': machine_idx, 'start_time': start_time, 'end_time': end_time, 'status': status, 'flag': flag})
                        visible_job_ids.add(job_id)
    sorted_job_ids = sorted(list(visible_job_ids))
    job_palette = cm.tab20(np.linspace(0, 1, len(sorted_job_ids)))
    colors_by_job = {job_id: job_palette[i] for i, job_id in enumerate(sorted_job_ids)}
    fig, ax = plt.subplots(figsize=(18, max(5, num_machines * 0.5)))
    column_names = schedule_rows[0].keys()
    with open('data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(schedule_rows)
    for record in schedule_rows:
        job_id = record['job_id']
        machine_idx = record['machine_id']
        start = record['start_time']
        end = record['end_time']
        processing_duration = end - start
        color = colors_by_job.get(job_id, 'grey')
        op_id = record['op_id']
        status = record['status']
        edge_color = 'black'
        alpha = 0.9
        hatch = None
        if status == 2:
            alpha = 0.6
        if processing_duration <= 1e-06:
            continue
        ax.barh(y=machine_idx, width=processing_duration, left=start, height=0.6, color=color, edgecolor=edge_color, alpha=alpha, hatch=hatch)
        label_x = start + processing_duration / 2
        label_y = machine_idx
        text_color = 'white' if np.mean(color[:3]) < 0.5 else 'black'
        if 'flag' in record:
            flag = record['flag']
            ax.text(label_x, label_y, f'J{job_id}-{flag}', ha='center', va='center', color=text_color, fontsize=6, fontweight='bold', clip_on=True)
        else:
            ax.text(label_x, label_y, f'J{job_id}', ha='center', va='center', color=text_color, fontsize=7, fontweight='bold', clip_on=True)
    max_time = max((record['end_time'] for record in schedule_rows if np.isfinite(record['end_time'])))
    if not np.isfinite(max_time):
        max_time = 100
    ax.set_xlim(0, max_time * 1.05)
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'M{i}' for i in range(num_machines)])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title(title, fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    if event_time > 0:
        plt.axvline(x=event_time, c='r', ls='-.', lw=2)
        plt.axvline(x=event_time + TIME_WINDOW_T_w, c='g', ls='-.', lw=2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        plt.savefig(filename, dpi=200)
    except Exception as e:
        warnings.warn(f'Error saving Gantt chart to {filename}: {e}')
    finally:
        plt.close(fig)
