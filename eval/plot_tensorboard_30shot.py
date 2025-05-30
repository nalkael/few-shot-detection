"""
show plot of curves from tensorboard format
@author: Huaixin Luo
"""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# load event file
event_file = "checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_30shot/events.out.tfevents.1745334365.PC-RD-343.14845.0"

event_acc = EventAccumulator(event_file)
event_acc.Reload() # load all data

print("Available metrics:", event_acc.Tags()['scalars'])

base_metrics = [
    'total_loss',
    'bbox/AP', 
    'bbox/AP50', 
    'bbox/AP75', 
    'bbox/APs', 
    'bbox/APm',
]

class_ap_metrcis = [
    'bbox/AP',
    'bbox/AP50', 
    'bbox/AP75',
    'bbox/APs', 
    'bbox/APm',
    # 'bbox/AP-Gasschieberdeckel', 
    # 'bbox/AP-Kanalschachtdeckel', 
    # 'bbox/AP-Sinkkaesten', 
    # 'bbox/AP-Unterflurhydrant', 
    # 'bbox/AP-Versorgungsschacht', 
    # 'bbox/AP-Wasserschieberdeckel',
]

desired_metrics = class_ap_metrcis

# Dictionary to store steps and values for each metric
metric_data = {}

# Custom labels for the metrics (shorten as needed)
custom_ap_labels = {
    'bbox/AP': 'mAP',
    'bbox/AP50': 'AP50',
    'bbox/AP75': 'AP75',
    'bbox/APs': 'APs',
    'bbox/APm': 'APm',
    # Example for per-class APs - adjust based on your class names
    # 'bbox/AP-Gasschieberdeckel': 'AP (Gas)',
    # 'bbox/AP-Kanalschachtdeckel': 'AP (Manhole)',
    # 'bbox/AP-Sinkkaesten': 'AP (Sink)',
    # 'bbox/AP-Unterflurhydrant': 'AP(Hydrant)', 
    # 'bbox/AP-Versorgungsschacht' : 'AP(Utility)', 
    # 'bbox/AP-Wasserschieberdeckel': 'AP(Water)',
    # Add more as needed based on your event file
}

# Extract data for each metric
for metric in desired_metrics:
    if metric in event_acc.Tags()['scalars']:
        steps = []
        values = []
        for event in event_acc.Scalars(metric):
            steps.append(event.step)
            values.append(event.value)
        metric_data[metric] = (steps, values)
    else:
        print(f"Warning: Metric '{metric}' not found in event file.")


# --------------------
# find highst mAP and AP50 with corresponding iterations
max_map = -1
max_map_iteration = None
max_ap50 = -1
max_ap50_iteration = None

if 'bbox/AP' in metric_data:
    steps, values = metric_data['bbox/AP']
    max_map = max(values)
    max_map_index = values.index(max_map)
    max_map_iteration = steps[max_map_index]

if 'bbox/AP50' in metric_data:
    steps, values = metric_data['bbox/AP50']
    max_ap50 = max(values)
    max_ap50_index = values.index(max_ap50)
    max_ap50_iteration = steps[max_ap50_index]

# Print the results
if max_map_iteration is not None:
    print(f"Highest mAP: {max_map:.4f} at iteration {max_map_iteration}")
else:
    print("mAP data not available.")

if max_ap50_iteration is not None:
    print(f"Highest AP50: {max_ap50:.4f} at iteration {max_ap50_iteration}")
else:
    print("AP50 data not available.")


# ---------------------------
# Plot all metrics in one figure with different colors
# Plot 1: AP Metrics
plt.figure(figsize=(16, 8))  # Wider figure: 16 inches wide, 8 inches tall
for metric, (steps, values) in metric_data.items():
    label = custom_ap_labels.get(metric, metric)
    plt.plot(steps, values, label=label)

plt.xlabel('Iteration', fontsize=24)  # Larger font size
plt.ylabel('Average Precision (AP)', fontsize=24)  # Larger font size
plt.title('Validation Metrics over Iterations', fontsize=18)  # Larger font size
plt.subplots_adjust(left=0.15, right=0.75, top=0.9, bottom=0.15)  # Adjusted margins
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=20)  # Larger legend font size
plt.grid(True)


# Save as SVG (optional)
plt.savefig('validation_ap_metrics.svg', format='svg', bbox_inches='tight')

# Display the plot
plt.show()

# --- Plot 2: Loss ---

loss_metrics = [
    'total_loss',
    'loss_cls',
    'loss_box_reg',
    ]

metric_data = {}

custom_loss_labels = {
    'total_loss' : 'Total Loss',
    'loss_cls': 'Class Loss',
    'loss_box_reg': 'Box Reg Loss',}

# Extract data for each metric
for metric in loss_metrics:
    if metric in event_acc.Tags()['scalars']:
        steps = []
        values = []
        for event in event_acc.Scalars(metric):
            steps.append(event.step)
            values.append(event.value)
        metric_data[metric] = (steps, values)
    else:
        print(f"Warning: Metric '{metric}' not found in event file.")

# Plot all metrics in one figure with different colors
plt.figure(figsize=(16, 8))  # Wider figure: 16 inches wide, 8 inches tall
for metric, (steps, values) in metric_data.items():
    label = custom_loss_labels.get(metric, metric)
    plt.plot(steps, values, label=label)

plt.xlabel('Iteration', fontsize=24)  # Larger font size
plt.ylabel('Loss', fontsize=24)  # Larger font size
plt.title('Training Metrics over Iterations', fontsize=18)  # Larger font size
plt.ylim(0.0, 1.0)
plt.subplots_adjust(left=0.15, right=0.75, top=0.9, bottom=0.15)  # Adjusted margins
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=20)  # Larger legend font size
plt.grid(True)

# Save as SVG (optional)
plt.savefig('training_loss_metrics.svg', format='svg', bbox_inches='tight')

# Display the plot
plt.show()
