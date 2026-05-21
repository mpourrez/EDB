import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from app_statistics import *
from utils import *
import seaborn as sns


######################################################################################################
######################################################################################################
######################################################################################################
def resource_to_value_list(resource_name, values):
    if resource_name == 'CPU-USER':
        return values.cpu_user
    elif resource_name == 'CPU-SYS':
        return values.cpu_sys
    elif resource_name == 'MEM':
        return values.mem
    elif resource_name == 'DISK':
        return values.disk
    elif resource_name == 'NET-REC':
        return values.net_rec
    elif resource_name == 'NET-TRA':
        return values.net_tra


def resource_to_plot_label(resource_name):
    if resource_name == 'CPU-USER':
        return 'CPU USER Utilization (%)'
    elif resource_name == 'CPU-SYS':
        return 'CPU System Utilization (%)'
    elif resource_name == 'MEM':
        return 'Memory Utilization (%)'
    elif resource_name == 'DISK':
        return 'Disk Utilization (%)'
    elif resource_name == 'NET-REC':
        return 'Network Received Packets'
    elif resource_name == 'NET-TRA':
        return 'Network Transmitted Packets'
    return "NO RESOURCE MATCHED"


######################################################################################################
######################################################################################################
######################################################################################################
def draw_heatmap(x_labels, y_labels, heatmap_data, edge_device, heatmap_title):
    plt.rcParams["font.family"] = "Times New Roman"
    # create a 2D array of data from the list of tuples
    array_data = np.zeros((len(y_labels), len(x_labels)))
    for item in heatmap_data:
        array_data[y_labels[item[1]], x_labels[item[0]]] = item[2]
    # create a heatmap with the data
    fig, ax = plt.subplots(figsize=(19, 10))  # Adjust the figsize to maximize space
    # heatmap = ax.pcolor(array_data, cmap='Reds')
    heatmap = ax.pcolor(array_data, cmap='Reds', vmin=0, vmax=90)
    # add a colorbar legend to the heatmap with percentage formatting
    cbar = plt.colorbar(heatmap, format='%.1f%%', pad=0.02)
    # add labels to the axes
    ax.set_xticks(np.arange(0.5, len(x_labels)))
    ax.set_yticks(np.arange(0.5, len(y_labels)))
    ax.set_xticklabels(list(x_labels.keys()), fontsize=18, rotation=90)
    ax.set_yticklabels(list(y_labels.keys()), fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18, direction='out', width=1, length=4)
    ax.tick_params(axis='both', which='minor', labelsize=18, direction='out', width=1, length=4)

    # Set the spine properties to make the subplot box border bold
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Set border width

    # increase the bottom margin of the plot
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(top=0.95)
    # fig.subplots_adjust(left=0.55)
    # fig.subplots_adjust(right=0.99)
    fig.suptitle(heatmap_title, fontsize=20, fontweight='bold')

    # Add y and x labels
    ax.set_ylabel('Fault Type', fontweight='bold', fontsize=20)
    ax.set_xlabel('Application-Resource Name', fontweight='bold', fontsize=20)

    # Customize colorbar tick label font properties
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontweight='bold', fontsize=18)
    cbar.set_label(label='Jitter Percentage on ' + edge_device, fontweight='bold', fontsize=20)

    # display the plot
    plt.tight_layout()
    plt.show()


######################################################################################################
######################################################################################################
######################################################################################################
def draw_fault_free_resource_comparisons(resource_name):
    app_statistics_rpi = {}
    app_statistics_nano = {}
    device_number = 2
    all_apps_rpi = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
                'IP', 'SA', 'PS', 'AE',  'OD-CPU', 'IC-A-CPU']
    all_apps_nano = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
                    'IP', 'SA', 'PS', 'AE',  'OD-CPU', 'OD-GPU']
    micro_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF']

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff5733', '#008000', '#800080']


    for app in all_apps_rpi:
        app_stat = AppFaultStatistics('raspberrypi' + str(device_number+1), app, 'No-Fault')
        app_statistics_rpi[app] = app_stat.get_resource_summary_matrix()

    for app in all_apps_nano:
        app_stat = AppFaultStatistics('nano' + str(device_number+1), app, 'No-Fault')
        app_statistics_nano[app] = app_stat.get_resource_summary_matrix()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for i, app in enumerate(all_apps_rpi):
        summary_matrix = app_statistics_rpi[app]
        utilizations = []
        times = []
        index = 0
        for key in summary_matrix.keys():
            value = summary_matrix[key]
            utilizations.append(resource_to_value_list(resource_name, value))
            times.append(index)
            index += 1
        times = times[0: len(times) - 15]
        utilizations = utilizations[0: len(utilizations) - 15]
        if app in micro_apps:
            axes[0][0].plot(times, utilizations, color=colors[i], label=app)
        else:
            axes[1][0].plot(times, utilizations, color=colors[i], label=app)

    for i, app in enumerate(all_apps_nano):
        summary_matrix = app_statistics_nano[app]
        utilizations = []
        times = []
        index = 0
        for key in summary_matrix.keys():
            value = summary_matrix[key]
            utilizations.append(resource_to_value_list(resource_name, value))
            times.append(index)
            index += 1
        times = times[0: len(times) - 15]
        utilizations = utilizations[0: len(utilizations) - 15]
        if i == 11:
            i = 12
        if app in micro_apps:
            axes[0][1].plot(times, utilizations, color=colors[i], label=app)
        else:
            axes[1][1].plot(times, utilizations, color=colors[i], label=app)

    plt.subplots_adjust(top=0.95)
    axes[0][0].set_ylabel(resource_to_plot_label(resource_name), fontsize=14, fontweight='bold')
    axes[0][1].set_ylabel(resource_to_plot_label(resource_name), fontsize=14, fontweight='bold')
    axes[1][0].set_ylabel(resource_to_plot_label(resource_name), fontsize=14, fontweight='bold')
    axes[1][1].set_ylabel(resource_to_plot_label(resource_name), fontsize=14, fontweight='bold')
    axes[0][0].set_title('Micro Benchmarks on Raspberry Pi', fontweight='bold')
    axes[0][1].set_title('Micro Benchmarks on Jetson Nano', fontweight='bold')
    axes[1][0].set_title('Application Benchmarks on Raspberry Pi', fontweight='bold')
    axes[1][1].set_title('Application Benchmarks on Jetson Nano', fontweight='bold')
    # axes[0].set_ylim(19, 27)
    # axes[1].set_ylim(19, 27)

    axises = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    for ax in axises:
        # ax.set_ylim(0,70)
        ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
        ax.tick_params(axis='both', which='minor', labelsize=14, width=2, length=4)
        # Set the spine properties to make the subplot box border bold
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # Set border width
        ax.legend()

    plt.tight_layout()
    plt.show()


######################################################################################################
######################################################################################################
######################################################################################################
def draw_all_faults_resource_comparisons(edge_device, micro_or_app, resource_name):
    app_statistics = {}
    micro_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF']
    macro_apps = ['IP', 'SA', 'IC-A-CPU', 'IC-S-CPU', 'OD-CPU', 'PS', 'AE']
    if micro_or_app == 'MICRO':
        all_apps = micro_apps
    else:
        all_apps = macro_apps
    faults = {'No-Fault': [0, 0], 'CPU-90': [0, 1], 'MEM-90%': [0, 2], 'IO-100': [1, 0], 'PF-0': [1, 1],
              'CCHE-0': [1, 2]}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for app in all_apps:
        for fault in faults.keys():
            app_stat = AppFaultStatistics(edge_device, app, fault)
            print('{0} - {1}'.format(app, fault))
            app_statistics[app + '-' + fault] = app_stat.get_resource_summary_matrix()

    for app in all_apps:
        for fault in faults.keys():
            print('Second Loop: {0} - {1}'.format(app, fault))
            summary_matrix = app_statistics[app + '-' + fault]
            utilizations = []
            times = []
            index = 0
            for key in summary_matrix.keys():
                value = summary_matrix[key]
                utilizations.append(resource_to_value_list(resource_name, value))
                times.append(index)
                index += 1
            axes[faults[fault][0]][faults[fault][1]].plot(times, utilizations, label=app)

    plt.subplots_adjust(top=0.95)
    for fault in faults.keys():
        axes[faults[fault][0]][faults[fault][1]].set_title(fault, fontweight='bold')
        axes[faults[fault][0]][faults[fault][1]].set_ylabel(resource_to_plot_label(resource_name), fontweight='bold')
        axes[faults[fault][0]][faults[fault][1]].legend()
    plt.show()


######################################################################################################
######################################################################################################
######################################################################################################
# Function to convert datetime objects to seconds relative to the start time
def convert_to_seconds(datetime_list, start_time):
    return [(dt - start_time).total_seconds() for dt in datetime_list]

def draw_single_app_all_resource_comp(edge_device, app, fault):
    device_number = 0
    if edge_device == 'Raspberry Pi':
        device_filename = 'raspberrypi'
    else:
        device_filename = 'nano'

    if device_filename == 'nano' and device_number == 1 and app == 'OD-CPU':
        return
    # faults = {'No-Fault':[0,0], 'CPU-90': [0,1], 'MEM-60%':[1,0], 'IO-100': [1,1], 'PF-0': [2,0], 'CCHE-0':[2,1]}
    resources = {'CPU-USER': [0, 0], 'CPU-SYS': [0, 1], 'MEM': [1, 0], 'DISK': [1, 1]}

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    print('{0} - {1}'.format(app, fault))
    app_stat = AppFaultStatistics(device_filename + str(device_number + 1), app, fault)

    for resource_name in resources:
        print('Second Loop: {0} - {1}'.format(app, fault))
        summary_matrix = app_stat.get_resource_summary_matrix()
        utilizations = []
        times = []
        # index = 0
        fault_injection_times = []
        fault_stop_times = []
        previous_label = 'No-Fault'
        for key in summary_matrix.keys():
            value = summary_matrix[key]
            utilizations.append(resource_to_value_list(resource_name, value))
            # times.append(index)
            # if previous_label == 'No-Fault' and value.label != 'No-Fault':
            #     fault_injection_times.append(index)
            # elif previous_label != 'No-Fault' and value.label == 'No-Fault':
            #     fault_stop_times.append(index)
            # index += 1
            # previous_label = value.label

        fault_start_times, fault_end_datetimes = app_stat.get_all_fault_injection_start_stop_times()
        start_time = app_stat.get_experiment_start_datetime()
        times = app_stat.cpu_datetimes
        ax = axes[resources[resource_name][0]][resources[resource_name][1]]
        if fault == 'No-Fault':
            ax.plot(convert_to_seconds(times[0:len(times) - 10], start_time), utilizations[0:len(utilizations) - 10])
        else:
            ax.plot(convert_to_seconds(times, start_time), utilizations)
        for (fault_start, fault_stop) in zip(convert_to_seconds(fault_start_times, start_time),
                                             convert_to_seconds(fault_end_datetimes, start_time)):
            ax.axvline(x=fault_start, color='r', linestyle='dashed')
            ax.axvline(x=fault_stop, color='b', linestyle='dashed')

        ax.set_title(app + '-' + fault + ' ' + edge_device + ' ' + str(device_number + 1), fontweight='bold',
                     fontsize=16)
        ax.set_ylabel(resource_to_plot_label(resource_name), fontweight='bold', fontsize=14)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Time', fontsize=14, fontweight='bold')
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
        ax.tick_params(axis='both', which='minor', labelsize=14, width=2, length=4)
        # Set the spine properties to make the subplot box border bold
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # Set border width
        if fault != 'No-Fault':
            ax.set_xlim(-30, 400)
        # ax.legend()

    # plt.subplots_adjust(top=0.95)
    # for fault in faults.keys():
    #     ax.set_title(app + '-' + fault, fontweight='bold')
    #     ax.set_ylabel(resource_to_plot_label(resource_name), fontweight='bold')
    #     ax.legend()
    plt.tight_layout()
    plt.show()
def draw_resource_comparisons_single_app_all_faults(edge_device, app, resource_name):
    device_number = 0
    app_statistics = {}
    if edge_device == 'Raspberry Pi':
        device_filename = 'raspberrypi'
    else:
        device_filename = 'nano'

    if device_filename == 'nano' and device_number == 1 and app == 'OD-CPU':
        return
    # faults = {'No-Fault':[0,0], 'CPU-90': [0,1], 'MEM-60%':[1,0], 'IO-100': [1,1], 'PF-0': [2,0], 'CCHE-0':[2,1]}
    faults = {'No-Fault':[0,0], 'MEM-20%': [0,1], 'MEM-60%':[1,0], 'IO-100': [1,1], 'PF-0': [2,0], 'CCHE-0':[2,1]}

    fig, axes = plt.subplots(3, 2, figsize=(20, 10))

    for fault in faults.keys():
        print('{0} - {1}'.format(app, fault))
        app_statistics[app + '-' + fault] = AppFaultStatistics(device_filename+str(device_number+1), app, fault)

    for fault in faults.keys():
        print('Second Loop: {0} - {1}'.format(app, fault))
        app_stat = app_statistics[app + '-' + fault]
        summary_matrix = app_stat.get_resource_summary_matrix()
        utilizations = []
        times = []
        # index = 0
        fault_injection_times = []
        fault_stop_times = []
        previous_label = 'No-Fault'
        for key in summary_matrix.keys():
            value = summary_matrix[key]
            utilizations.append(resource_to_value_list(resource_name, value))
            # times.append(index)
            # if previous_label == 'No-Fault' and value.label != 'No-Fault':
            #     fault_injection_times.append(index)
            # elif previous_label != 'No-Fault' and value.label == 'No-Fault':
            #     fault_stop_times.append(index)
            # index += 1
            # previous_label = value.label

        fault_start_times, fault_end_datetimes = app_stat.get_all_fault_injection_start_stop_times()
        start_time = app_stat.get_experiment_start_datetime()
        times = app_stat.cpu_datetimes
        ax = axes[faults[fault][0]][faults[fault][1]]
        if fault == 'No-Fault':
            ax.plot(convert_to_seconds(times[0:len(times)-10], start_time), utilizations[0:len(utilizations)-10])
        else:
            ax.plot(convert_to_seconds(times, start_time), utilizations)
        for (fault_start, fault_stop) in zip(convert_to_seconds(fault_start_times, start_time),
                                                                convert_to_seconds(fault_end_datetimes, start_time)):
            ax.axvline(x=fault_start, color='r', linestyle='dashed')
            ax.axvline(x=fault_stop, color='b', linestyle='dashed')

        ax.set_title(app + '-' + fault + ' ' + edge_device+' '+str(device_number+1), fontweight='bold', fontsize=16)
        ax.set_ylabel(resource_to_plot_label(resource_name), fontweight='bold', fontsize=14)
        ax.set_xlabel('Time', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
        ax.tick_params(axis='both', which='minor', labelsize=14, width=2, length=4)
        # Set the spine properties to make the subplot box border bold
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # Set border width
        if fault != 'No-Fault':
            ax.set_xlim(-30, 200)
        # ax.legend()

    # plt.subplots_adjust(top=0.95)
    # for fault in faults.keys():
    #     ax.set_title(app + '-' + fault, fontweight='bold')
    #     ax.set_ylabel(resource_to_plot_label(resource_name), fontweight='bold')
    #     ax.legend()
    plt.tight_layout()
    plt.show()


######################################################################################################
######################################################################################################
######################################################################################################

def draw_single_utilization(edge_device, application, fault, resource_name):
    if edge_device == 'Raspberry Pi':
        edge_device_filename = 'raspberrypi'
    else:
        edge_device_filename = 'nano'
    app_statistics = AppFaultStatistics(edge_device_filename+'1', application, fault)
    resource_summary_matrix = app_statistics.get_resource_summary_matrix()
    utilizations = []
    times = []
    fault_injection_times = []
    fault_stop_times = []
    previous_label = 'No-Fault'
    for key in resource_summary_matrix.keys():
        value = resource_summary_matrix[key]
        utilizations.append(resource_to_value_list(resource_name, value))
        if previous_label == 'No-Fault' and value.label != 'No-Fault':
            fault_injection_times.append(key)
        elif previous_label != 'No-Fault' and value.label == 'No-Fault':
            fault_stop_times.append(key)
        times.append(key)
        previous_label = value.label

    print(fault_injection_times)
    print(fault_stop_times)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    axs.plot(times, utilizations, label='memused')
    for (fault_start, fault_stop) in zip(fault_injection_times, fault_stop_times):
        axs.axvline(x=fault_start, color='r', linestyle='dashed')
        axs.axvline(x=fault_stop, color='b', linestyle='dashed')

    axs.set_xlabel('Time', fontsize=16, fontweight='bold')
    axs.set_ylabel(resource_to_plot_label(resource_name), fontsize=16, fontweight='bold')
    axs.grid(which="major")
    fig.suptitle(' Utilization Plot for App: ' + application + ' - Fault: ' + fault,
                 fontsize=16, fontweight='bold')
    plt.show()


######################################################################################################
######################################################################################################
######################################################################################################
def draw_latency_all_apps_all_faults(edge_device):
    all_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
                'IP', 'SA', 'PS', 'AE', 'OD-CPU', 'IC-A-CPU', 'IC-S-CPU']
    faults = ['No-Fault', 'CPU-20', 'CPU-60', 'CPU-90', 'MEM-20%', 'MEM-60%',
              'MEM-90%', 'IO-100', 'PF-0', 'CCHE-0', 'PING', 'TCP']

    app_statistics = {}
    for app in all_apps:
        for fault in faults:
            print(app + '-' + fault)
            app_statistics[app + '-' + fault] = AppFaultStatistics(edge_device, app, fault)

    fig, axes = plt.subplots(1, 1, figsize=(12, 5))

    xlabels = []
    for app in all_apps:
        for fault in faults:
            print(f"*******{app}-{fault}*****")
            computation_times, _ = app_statistics[app + '-' + fault].get_all_latencies()
            xlabel = '' + app + '-' + fault
            xlabels.append(xlabel)
            axes.bar(x=xlabel, height=get_avg_without_outlier(computation_times),
                     yerr=get_std_without_outlier(computation_times))

    axes.set_title('Computation Time', fontweight='bold')
    axes.set_ylabel('AVG Computation Time (ms)', fontweight='bold')
    plt.subplots_adjust(top=0.95)
    # Rotate x-axis labels for all subplots

    axes.set_xticklabels(xlabels, rotation=90)
    # Adjust layout and display
    plt.tight_layout()
    plt.show()


######################################################################################################
######################################################################################################
######################################################################################################
def get_faulty_values(fi_starts, fi_stops, request_received_t, response_ready_t, latencies):
    # return latencies
    faulty_values = []
    for (req_t, res_t, latency) in zip(request_received_t, response_ready_t, latencies):
        for (start, stop) in zip(fi_starts, fi_stops):
            if req_t>start and req_t<stop:
                faulty_values.append(latency)
    return faulty_values


def draw_latency_all_apps_subfigures(device_name, micros_or_macros, latency_type):
    max_edge_devices = 5
    if micros_or_macros == 'micros':
        all_apps = ['FFT']
        # all_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF']
    else:
        if device_name == 'Raspberry Pi':
            all_apps = ['IP', 'SA', 'PS', 'AE', 'OD-CPU', 'IC-A-CPU']
        else:
            all_apps = ['IP', 'SA', 'PS', 'AE', 'OD-CPU']

    # faults = ['No-Fault', 'CPU-20', 'CPU-60', 'CPU-90', 'MEM-20%', 'MEM-60%',
    #           'MEM-90%', 'IO-100', 'PF-0', 'CCHE-0', 'PING', 'TCP']
    faults = ['No-Fault', 'MEM-20%',  'IO-100']
    if device_name == 'Raspberry Pi':
        device_filename = 'raspberrypi'
    else:
        device_filename = 'nano'
        # faults = ['No-Fault', 'CPU-20', 'CPU-60', 'CPU-90', 'MEM-20%', 'MEM-60%',
        #           'IO-100', 'PF-0', 'CCHE-0', 'PING', 'TCP']

    app_statistics = {}
    for app in all_apps:
        for fault in faults:
            for device_number in range(4, max_edge_devices):
                if device_filename == 'nano' and device_number == 0 and app == 'SORT' and fault == 'CCHE-0':
                    continue
                if device_filename == 'nano' and device_number == 1 and app == 'AE' and (fault == 'CCHE-0' or fault=='IO-100' or fault=='PF-0' or fault=='CPU-90' or 'MEM' in fault):
                    continue
                print(device_filename + str(device_number + 1) + ':' + app + '-' + fault)
                app_statistics[app + '-' + fault + '-' + str(device_number)] = \
                    AppFaultStatistics(device_filename + str(device_number + 1), app, fault)

    # Set the font family to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    # Set the CUD palette
    cud_palette = sns.color_palette("colorblind", n_colors=12)
    hatch_patterns = ['', '\\\\\\', 'xxx', '---', '////', '\\\\\\\\', '....',
                      '++++', '||||', 'xxxx', '////', '\\\\\\\\']

    fault_styles = {}
    for (i, fault) in zip(range(len(faults)), faults):
        # if device_filename=='nano' and \
        #         (fault == 'IO-100' or fault == 'PF-0' or fault == 'CCHE-0' or fault == 'TCP' or fault == 'PING'):
        #     i += 1
        fault_styles[fault] = {'color': cud_palette[i], 'hatch': hatch_patterns[i]}

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    # Reduce space between subplots
    # grid = GridSpec(2, 3, figure=fig, wspace=0.1, hspace=0.2)  # Adjust the values as needed

    row = 0
    col = 0
    for app in all_apps:
        for fault in faults:
            rtt_faulties = []
            computation_times_faulties = []
            transmission_times_faulties = []
            for device_number in range(4, max_edge_devices):
                if device_filename == 'nano' and device_number == 0 and app == 'SORT' and fault == 'CCHE-0':
                    continue
                if device_filename == 'nano' and device_number == 1 and app == 'AE' and (
                        fault == 'CCHE-0' or fault == 'IO-100' or fault == 'PF-0' or fault == 'CPU-90' or 'MEM' in fault):
                    continue
                print(f"*******{device_filename} {device_number}: {app}-{fault}*****")
                request_received_time, response_ready_time, rtts, computation_times, transmission_times = \
                    app_statistics[app + '-' + fault + '-' + str(device_number)].get_all_latencies_and_datetimes()
                fault_injection_start_time, fault_injection_stop_time = \
                    app_statistics[app + '-' + fault + '-' + str(device_number)].get_all_fault_injection_start_stop_times()
                if fault == 'No-Fault' or fault == 'PING' or fault == 'TCP':
                    rtt_faulties.extend(rtts)
                    computation_times_faulties.extend(computation_times)
                    transmission_times_faulties.extend(transmission_times)
                else:
                    rtt_faulties.extend(get_faulty_values(fault_injection_start_time, fault_injection_stop_time,
                                                          request_received_time, response_ready_time, rtts))
                    computation_times_faulties.extend(get_faulty_values(fault_injection_start_time, fault_injection_stop_time,
                                                                        request_received_time, response_ready_time, computation_times))
                    transmission_times_faulties.extend(get_faulty_values(fault_injection_start_time, fault_injection_stop_time,
                                                                         request_received_time, response_ready_time, transmission_times))

            rtt_faulties = [x / 1000 for x in rtt_faulties]
            computation_times_faulties = [x / 1000 for x in computation_times_faulties]
            transmission_times_faulties = [x / 1000 for x in transmission_times_faulties]
            error_kw = {'elinewidth': 2}
            ax = axes[row][col]
            style = fault_styles[fault]
            if latency_type == 'computation':
                ax.bar(x=fault, height=get_avg_without_outlier(computation_times_faulties),
                       yerr=get_std_without_outlier(computation_times_faulties),
                       color=style['color'], hatch=style['hatch'], error_kw=error_kw)
                ax.set_title(app + ' Computation Time on ' + edge_device, fontweight='bold', fontsize=18)
                ax.set_ylabel('Average Computation Time (s)', fontweight='bold', fontsize=18)
            elif latency_type == 'rtt':
                ax.bar(x=fault, height=get_avg_without_outlier(rtt_faulties),
                       yerr=get_std_without_outlier(rtt_faulties),
                       color=style['color'], hatch=style['hatch'], error_kw=error_kw)
                ax.set_title(app + ' Response Time on ' + edge_device, fontweight='bold', fontsize=18)
                ax.set_ylabel('Average Response Time (s)', fontweight='bold', fontsize=18)
            else:
                ax.bar(x=fault, height=get_avg_without_outlier(transmission_times_faulties),
                       yerr=get_std_without_outlier(transmission_times_faulties),
                       color=style['color'], hatch=style['hatch'], error_kw=error_kw)
                ax.set_title(app + ' Transmission Time on ' + edge_device, fontweight='bold', fontsize=18)
                ax.set_ylabel('Average Transmission Time (s)', fontweight='bold', fontsize=18)
            ax.set_xticklabels(faults, rotation=90)
            ax.set_xlabel('')  # Clear the x-axis label
            ax.tick_params(axis='both', which='major', labelsize=18, direction='out', width=1, length=2)
            ax.tick_params(axis='both', which='minor', labelsize=18, direction='out', width=1, length=2)

            # Set the spine properties to make the subplot box border bold
            for spine in ax.spines.values():
                spine.set_linewidth(2)  # Set border width

            # Add grid lines inside the figure
            ax.grid(which='both', linestyle='--', linewidth=1)
        col += 1
        if col == 3:
            row += 1
            col = 0
    # plt.subplots_adjust(top=1)
    plt.tight_layout()
    plt.show()


######################################################################################################
######################################################################################################
######################################################################################################

def draw_latency_single_app_over_time(edge_device, app):
    app_statistics = {}
    start_times = {}
    if edge_device == 'Raspberry Pi':
        device_filename = 'raspberrypi'
    else:
        device_filename = 'nano'
    faults = {'No-Fault':[0,0], 'CPU-90': [0,1], 'MEM-60%':[1,0], 'IO-100': [1,1], 'PF-0': [2,0], 'CCHE-0':[2,1]}

    device_number = 0
    for fault in faults:
        if device_filename == 'nano' and device_number == 1 and app == 'OD-CPU' and fault == 'MEM-60%':
            continue
        print(f"********Reading: {fault} - Device Number: {device_number}")
        app_stat = AppFaultStatistics(device_filename+str(device_number+1), app, fault)
        app_statistics[fault+'-'+str(device_number)] = app_stat
        start_times[fault + '-' + str(device_number)] = app_stat.get_experiment_start_datetime()

    fig, axes = plt.subplots(3, 2, figsize=(20, 10))

    for fault in faults.keys():
        rtts = []
        request_received_times = []
        fault_injection_start_times = []
        fault_injection_stop_times = []

        start_time = start_times[fault + '-' + str(device_number)]
        request_received_time, response_ready_time, rtt, computation_times, transmission_times = \
            app_statistics[fault + '-' + str(device_number)].get_all_latencies_and_datetimes()
        fault_injection_start_time, fault_injection_stop_time = \
            app_statistics[fault + '-' + str(device_number)].get_all_fault_injection_start_stop_times()
        rtts.extend(rtt)
        request_received_times.extend(request_received_time)
        fault_injection_start_times.extend(fault_injection_start_time)
        fault_injection_stop_times.extend(fault_injection_stop_time)

        rtts = [x / 1000 for x in rtts]
        ax = axes[faults[fault][0]][faults[fault][1]]
        print(len(rtts))
        ax.plot(convert_to_seconds(request_received_times, start_time), rtts, label='rtt')
        xs = convert_to_seconds(request_received_times, start_time)
        xmax = max(xs) + 5
        xmin = min(xs) - 5
        ax.set_xlim(0, xmax)
        if fault != 'No-Fault':
            # for (fault_start, fault_stop) in zip(convert_to_seconds(fault_injection_start_times, start_time),
            #                                      convert_to_seconds(fault_injection_stop_times, start_time)):
            #     ax.axvline(x=fault_start, color='r', linestyle='dashed')
            #     ax.axvline(x=fault_stop, color='b', linestyle='dashed')
            for fs, fe in zip(convert_to_seconds(fault_injection_start_times, start_time),
                              convert_to_seconds(fault_injection_stop_times, start_time)):
                ax.axvspan(fs, fe, alpha=0.12, color='tab:red', zorder=0)  # shaded window
                ax.axvline(fs, color='tab:red', linestyle='--', linewidth=2, zorder=1)
                ax.axvline(fe, color='tab:blue', linestyle='--', linewidth=2, zorder=1)
        ax.set_title('Response Time under ' + fault + ' on ' + edge_device + ' ' + str(device_number+1),
                     fontweight='bold', fontsize=16)
        ax.set_ylabel('AVG Response Time (s)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Time', fontsize=14, fontweight='bold')
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
        ax.tick_params(axis='both', which='minor', labelsize=14, width=2, length=4)
        # Set the spine properties to make the subplot box border bold
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # Set border width
        ax.set_ylim(0, 2)
        if fault != 'No-Fault':
            ax.set_xlim(0, 200)

    plt.tight_layout()
    plt.show()

######################################################################################################
######################################################################################################
######################################################################################################
def draw_latency_single_app_all_faults(edge_device, app):
    app_statistics = {}
    if edge_device == 'Raspberry Pi':
        device_filename = 'raspberrypi'
    else:
        device_filename = 'nano'
    faults = ['No-Fault', 'CPU-20', 'CPU-60', 'CPU-90', 'MEM-20%', 'MEM-60%', 'MEM-90%', 'IO-100', 'PF-0', 'CCHE-0']

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for fault in faults:
        app_stat = AppFaultStatistics(device_filename+'1', app, fault)
        app_statistics[fault] = app_stat.get_resource_summary_matrix()

    for fault in faults:
        summary_matrix = app_statistics[fault]

        fault_free_comp_t = []
        faulty_comp_t = []

        fault_free_transmit_t = []
        faulty_transmit_t = []

        fault_free_rtt = []
        faulty_rtt = []

        for key in summary_matrix.keys():
            value = summary_matrix[key]
            if value.com_t is not None:
                if value.label == 'No-Fault':
                    fault_free_comp_t.append(value.com_t)
                    fault_free_transmit_t.append(value.tra_t)
                    fault_free_rtt.append(value.tra_t + value.com_t)
                else:
                    faulty_comp_t.append(value.com_t)
                    faulty_transmit_t.append(value.tra_t)
                    faulty_rtt.append(value.tra_t + value.com_t)

        if fault == 'No-Fault':
            # print(get_avg_without_outlier(fault_free_rtt))
            # print(get_std_without_outlier(fault_free_rtt))
            axes[0].bar(x=fault, height=get_avg_without_outlier(fault_free_rtt),
                        yerr=get_std_without_outlier(fault_free_rtt))
            axes[1].bar(x=fault, height=get_avg_without_outlier(fault_free_comp_t),
                        yerr=get_std_without_outlier(fault_free_comp_t))
            axes[2].bar(x=fault, height=get_avg_without_outlier(fault_free_transmit_t),
                        yerr=get_std_without_outlier(fault_free_transmit_t))
        else:
            axes[0].bar(x=fault, height=get_avg_without_outlier(faulty_rtt), yerr=get_std_without_outlier(faulty_rtt))
            axes[1].bar(x=fault, height=get_avg_without_outlier(faulty_comp_t),
                        yerr=get_std_without_outlier(faulty_comp_t))
            axes[2].bar(x=fault, height=get_avg_without_outlier(faulty_transmit_t),
                        yerr=get_std_without_outlier(faulty_transmit_t))

        axes[0].set_title(app + '-Response Time', fontweight='bold')
        axes[0].set_ylabel('AVG Response Time (ms)', fontweight='bold')

        axes[1].set_title(app + '-Computation Time', fontweight='bold')
        axes[1].set_ylabel('AVG Computation Time (ms)', fontweight='bold')

        axes[2].set_title(app + '-Transmission Time', fontweight='bold')
        axes[2].set_ylabel('AVG Transmission Time (ms)', fontweight='bold')

    plt.subplots_adjust(top=0.95)
    # Rotate x-axis labels for all subplots
    for ax in axes:
        ax.set_xticklabels(faults, rotation=45, ha='right')
    # Adjust layout and display
    plt.tight_layout()
    plt.show()


######################################################################################################
######################################################################################################
######################################################################################################
def draw_heatmaps_on_latency_jitter(edge_device):
    app_statistics = {}
    all_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
                'IP', 'SA', 'IC-A-CPU', 'IC-S-CPU', 'OD-CPU', 'PS', 'AE']
    faults = ['No-Fault', 'CPU-20', 'CPU-60', 'CPU-90', 'MEM-20%', 'MEM-60%', 'MEM-90%', 'IO-100', 'PF-0', 'CCHE-0']

    for app in all_apps:
        for fault in faults:
            app_stat = AppFaultStatistics(edge_device, app, fault)
            print(app + '-' + fault)
            app_statistics[app + '-' + fault] = app_stat.get_resource_summary_matrix()

    latency_jitter_heatmap_data = []
    for app in all_apps:
        for fault in faults:
            summary_matrix = app_statistics[app + '-' + fault]
            fault_free_comp_t = []
            faulty_comp_t = []

            fault_free_transmit_t = []
            faulty_transmit_t = []

            fault_free_rtt = []
            faulty_rtt = []

            for key in summary_matrix.keys():
                value = summary_matrix[key]
                if value.com_t is not None:
                    if value.label == 'No-Fault':
                        fault_free_comp_t.append(value.com_t)
                        fault_free_transmit_t.append(value.tra_t)
                        fault_free_rtt.append(value.tra_t + value.com_t)
                    else:
                        faulty_comp_t.append(value.com_t)
                        faulty_transmit_t.append(value.tra_t)
                        faulty_rtt.append(value.tra_t + value.com_t)
            if fault == 'No-Fault':
                latency_jitter_heatmap_data.append([(app + '-RTT'), fault, jitter(fault_free_rtt)])
                latency_jitter_heatmap_data.append([(app + '-COMP-T'), fault, jitter(fault_free_comp_t)])
                latency_jitter_heatmap_data.append([(app + '-TRAN-T'), fault, jitter(fault_free_transmit_t)])
            else:
                latency_jitter_heatmap_data.append([(app + '-RTT'), fault, jitter(faulty_rtt)])
                latency_jitter_heatmap_data.append([(app + '-COMP-T'), fault, jitter(faulty_comp_t)])
                latency_jitter_heatmap_data.append([(app + '-TRAN-T'), fault, jitter(faulty_transmit_t)])

    draw_heatmap(heatmap_latency_x_labels, heatmap_y_labels, latency_jitter_heatmap_data,
                 'Impact of Faults on Latency Jitter Heatmap')


######################################################################################################
######################################################################################################
######################################################################################################
def draw_heatmaps_on_latency_increase(edge_device):
    app_statistics = {}
    all_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
                'IP', 'SA', 'IC-A-CPU', 'IC-S-CPU', 'OD-CPU', 'PS', 'AE']
    faults = ['CPU-20', 'CPU-60', 'CPU-90', 'MEM-20%', 'MEM-60%', 'MEM-90%', 'IO-100', 'PF-0', 'CCHE-0']

    for app in all_apps:
        for fault in faults:
            app_stat = AppFaultStatistics(edge_device, app, fault)
            print(app + '-' + fault)
            app_statistics[app + '-' + fault] = app_stat.get_resource_summary_matrix()

    latency_increase_heatmap_data = []
    for app in all_apps:
        for fault in faults:
            summary_matrix = app_statistics[app + '-' + fault]
            fault_free_comp_t = []
            faulty_comp_t = []

            fault_free_transmit_t = []
            faulty_transmit_t = []

            fault_free_rtt = []
            faulty_rtt = []

            for key in summary_matrix.keys():
                value = summary_matrix[key]
                if value.com_t is not None:
                    if value.label == 'No-Fault':
                        fault_free_comp_t.append(value.com_t)
                        fault_free_transmit_t.append(value.tra_t)
                        fault_free_rtt.append(value.tra_t + value.com_t)
                    else:
                        faulty_comp_t.append(value.com_t)
                        faulty_transmit_t.append(value.tra_t)
                        faulty_rtt.append(value.tra_t + value.com_t)

            latency_increase_heatmap_data.append([(app + '-RTT'), fault,
                                                  math.fabs(get_avg_without_outlier(fault_free_rtt) -
                                                            get_avg_without_outlier(faulty_rtt))])
            latency_increase_heatmap_data.append([(app + '-COMP-T'), fault,
                                                  math.fabs(get_avg_without_outlier(fault_free_comp_t) -
                                                            get_avg_without_outlier(faulty_comp_t))])
            latency_increase_heatmap_data.append([(app + '-TRAN-T'), fault,
                                                  math.fabs(get_avg_without_outlier(fault_free_transmit_t) -
                                                            get_avg_without_outlier(faulty_transmit_t))])

    draw_heatmap(heatmap_latency_x_labels, heatmap_y_labels, latency_increase_heatmap_data,
                 'Impact of Faults on Average Latency Heatmap')


######################################################################################################
######################################################################################################
######################################################################################################
def draw_heatmaps_on_latency_increase_maybe_better_version(edge_device):
    app_statistics = {}
    all_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
                'IP', 'SA', 'IC-A-CPU', 'IC-S-CPU', 'OD-CPU', 'PS', 'AE']
    faults = ['No-Fault', 'CPU-20', 'CPU-60', 'CPU-90', 'MEM-20%', 'MEM-60%', 'MEM-90%', 'IO-100', 'PF-0', 'CCHE-0']

    for app in all_apps:
        for fault in faults:
            print(app + '-' + fault)
            app_statistics[app + '-' + fault] = AppFaultStatistics(edge_device, app, fault)

    latency_increase_heatmap_data = []
    fault_free_comp_times = []
    fault_free_tran_times = []
    for app in all_apps:
        for fault in faults:
            print(f"*******{app}-{fault}*****")
            computation_times, transmission_times = app_statistics[app + '-' + fault].get_all_latencies()
            if fault == 'No-Fault':
                fault_free_comp_times = computation_times
                fault_free_tran_times = transmission_times
            else:
                latency_increase_heatmap_data.append([(app + '-COMP-T'), fault,
                                                      math.fabs(get_avg_without_outlier(fault_free_comp_times) -
                                                                get_avg_without_outlier(computation_times))])
                # latency_increase_heatmap_data.append([(app + '-TRAN-T'), fault,
                #                                        math.fabs(get_avg_without_outlier(fault_free_tran_times) -
                #                                                  get_avg_without_outlier(transmission_times))])

    draw_heatmap(heatmap_latency_x_labels, heatmap_y_labels, latency_increase_heatmap_data,
                 'Impact of Faults on Average Latency Heatmap')


######################################################################################################
######################################################################################################
######################################################################################################
def draw_heatmaps_on_resource_increase(edge_device):
    max_edge_devices = 3
    if edge_device == 'Raspberry Pi':
        edge_device_filename = 'raspberrypi'
    else:
        edge_device_filename = 'nano'

    all_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
                'IP', 'SA', 'PS', 'AE', 'OD-CPU', 'IC-A-CPU']
    faults = ['CPU-20', 'CPU-60', 'CPU-90', 'MEM-20%', 'MEM-60%', 'MEM-90%', 'IO-100', 'PF-0', 'CCHE-0']
    if edge_device_filename == 'nano':
        faults = ['CPU-20', 'CPU-60', 'CPU-90', 'MEM-20%', 'MEM-60%', 'IO-100', 'PF-0', 'CCHE-0']
        all_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
                    'IP', 'SA', 'PS', 'AE', 'OD-CPU', 'OD-GPU']
    resource_types = ['CPU-USER', 'CPU-SYS', 'MEM', 'DISK']

    heatmap_x_labels = {}
    heatmap_y_labels = {}
    counter = 0
    for resource_type in resource_types:
        for app in all_apps:
            heatmap_x_labels[app + '-' + resource_type] = counter
            counter += 1

    counter = 0
    for fault in faults:
        heatmap_y_labels[fault] = counter
        counter += 1

    app_statistics = {}
    for app in all_apps:
        for fault in faults:
            for device_number in range(max_edge_devices):
                if edge_device_filename == 'nano' and device_number == 1 and app == 'OD-CPU' and fault == 'MEM-60%':
                    continue
                print(edge_device_filename + str(device_number + 1) + ':' + app + '-' + fault)
                app_stat = AppFaultStatistics(edge_device_filename + str(device_number + 1), app, fault)
                app_statistics[app + '-' + fault + '-' + str(device_number)] = app_stat.get_resource_summary_matrix()

    resource_increase_heatmap_data = []
    for app in all_apps:
        for fault in faults:
            print('*********' + app + '-' + fault)
            faulty_cpu_usr = []
            faulty_cpu_sys = []
            faulty_mem = []
            faulty_disk = []
            faulty_net_tra = []
            faulty_net_rec = []

            fault_free_cpu_usr = []
            fault_free_cpu_sys = []
            fault_free_mem = []
            fault_free_disk = []
            fault_free_net_tra = []
            fault_free_net_rec = []

            for device_number in range(max_edge_devices):
                if edge_device_filename == 'nano' and device_number == 1 and app == 'OD-CPU' and fault == 'MEM-60%':
                    continue
                summary_matrix = app_statistics[app + '-' + fault + '-' + str(device_number)]
                for key in summary_matrix.keys():
                    value = summary_matrix[key]
                    cpu_usr_value = resource_to_value_list('CPU-USER', value)
                    cpu_sys_value = resource_to_value_list('CPU-SYS', value)
                    mem_value = resource_to_value_list('MEM', value)
                    disk_value = resource_to_value_list('DISK', value)
                    net_tra_value = resource_to_value_list('NET-TRA', value)
                    net_rec_value = resource_to_value_list('NET-REC', value)

                    if value.label != 'No-Fault':
                        if cpu_usr_value is not None:
                            faulty_cpu_usr.append(cpu_usr_value)
                        if cpu_sys_value is not None:
                            faulty_cpu_sys.append(cpu_sys_value)
                        if mem_value is not None:
                            faulty_mem.append(mem_value)
                        if disk_value is not None:
                            faulty_disk.append(disk_value)
                        if net_tra_value is not None:
                            faulty_net_tra.append(net_tra_value)
                        if net_rec_value is not None:
                            faulty_net_rec.append(net_rec_value)
                    else:
                        if cpu_usr_value is not None:
                            fault_free_cpu_usr.append(cpu_usr_value)
                        if cpu_sys_value is not None:
                            fault_free_cpu_sys.append(cpu_sys_value)
                        if mem_value is not None:
                            fault_free_mem.append(mem_value)
                        if disk_value is not None:
                            fault_free_disk.append(disk_value)
                        if net_tra_value is not None:
                            fault_free_net_tra.append(net_tra_value)
                        if net_rec_value is not None:
                            fault_free_net_rec.append(net_rec_value)

            fault_free_cpu_usr = fault_free_cpu_usr[0: len(fault_free_cpu_usr) - 15]
            fault_free_cpu_sys = fault_free_cpu_sys[0: len(fault_free_cpu_sys) - 15]
            fault_free_mem = fault_free_mem[0: len(fault_free_mem) - 15]
            fault_free_disk = fault_free_disk[0: len(fault_free_disk) - 15]

            # print("CPU-USR")
            # resource_increase_heatmap_data.append([(app + '-CPU-USER'), fault,
            #                                        math.fabs((sum(fault_free_cpu_usr)/len(fault_free_cpu_usr)) -
            #                                                  (sum(faulty_cpu_usr)/len(faulty_cpu_usr)))])
            # print("CPU-SYS")
            # resource_increase_heatmap_data.append([(app + '-CPU-SYS'), fault,
            #                                        math.fabs((sum(fault_free_cpu_sys)/len(fault_free_cpu_sys)) -
            #                                                  (sum(faulty_cpu_sys)/len(faulty_cpu_sys)))])
            # print("MEM")
            # resource_increase_heatmap_data.append([(app + '-MEM'), fault,
            #                                        math.fabs((sum(fault_free_mem)/len(fault_free_mem)) -
            #                                                  (sum(faulty_mem)/len(faulty_mem)))])
            # print("CPU-DISK-{0}-{1}".format(len(fault_free_disk), len(faulty_disk)))
            # resource_increase_heatmap_data.append([(app + '-DISK'), fault,
            #                                        math.fabs((sum(fault_free_disk)/len(fault_free_disk)) -
            #                                                  (sum(faulty_disk)/len(faulty_disk)))])
            print("CPU-USR")
            resource_increase_heatmap_data.append([(app + '-CPU-USER'), fault,
                                                   math.fabs(get_avg_without_outlier(fault_free_cpu_usr) -
                                                             get_avg_without_outlier(faulty_cpu_usr))])
            print("CPU-SYS")
            resource_increase_heatmap_data.append([(app + '-CPU-SYS'), fault,
                                                   math.fabs(get_avg_without_outlier(fault_free_cpu_sys) -
                                                             get_avg_without_outlier(faulty_cpu_sys))])
            print("MEM")
            resource_increase_heatmap_data.append([(app + '-MEM'), fault,
                                                   math.fabs(get_avg_without_outlier(fault_free_mem) -
                                                             get_avg_without_outlier(faulty_mem))])
            print("CPU-DISK-{0}-{1}".format(len(fault_free_disk), len(faulty_disk)))
            resource_increase_heatmap_data.append([(app + '-DISK'), fault,
                                                   math.fabs(get_avg_without_outlier(fault_free_disk) -
                                                             get_avg_without_outlier(faulty_disk))])
            print("-------------------------------------------------------------------")

    draw_heatmap(heatmap_x_labels, heatmap_y_labels, resource_increase_heatmap_data, edge_device,
                 'Impact of Faults on Resource Utilization Increase Heatmap on ' + edge_device)


######################################################################################################
######################################################################################################
######################################################################################################
def draw_heatmaps_on_resource_jitter(edge_device):
    max_edge_devices = 3
    if edge_device == 'Raspberry Pi':
        edge_device_filename = 'raspberrypi'
    else:
        edge_device_filename = 'nano'

    all_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
                'IP', 'PS', 'AE', 'OD-CPU', 'IC-A-CPU']
    # all_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
    #             'IP', 'SA', 'PS', 'AE', 'OD-CPU', 'IC-A-CPU']
    faults = ['No-Fault', 'CPU-20', 'CPU-60', 'CPU-90', 'MEM-20%', 'MEM-60%', 'MEM-90%', 'IO-100', 'PF-0', 'CCHE-0']
    if edge_device_filename == 'nano':
        faults = ['No-Fault', 'CPU-20', 'CPU-60', 'CPU-90', 'MEM-20%', 'MEM-60%', 'IO-100', 'PF-0']#, 'CCHE-0']
        all_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
                    'IP', 'SA', 'PS', 'AE', 'OD-CPU']
        # all_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
        #             'IP', 'SA', 'PS', 'OD-CPU']
    resource_types = ['CPU-USER', 'CPU-SYS', 'MEM', 'DISK']

    heatmap_x_labels = {}
    heatmap_y_labels = {}
    counter = 0
    for resource_type in resource_types:
        for app in all_apps:
            heatmap_x_labels[app + '-' + resource_type] = counter
            counter += 1

    counter = 0
    for fault in faults:
        heatmap_y_labels[fault] = counter
        counter += 1

    app_statistics = {}
    for app in all_apps:
        for fault in faults:
            for device_number in range(max_edge_devices):
                if edge_device_filename == 'raspberrypi' and device_number == 0:
                    continue
                if edge_device_filename == 'nano' and device_number == 1 and app == 'OD-CPU' and fault == 'MEM-60%':
                    continue
                print(edge_device_filename + str(device_number + 1) + ':' + app + '-' + fault)
                app_stat = AppFaultStatistics(edge_device_filename + str(device_number + 1), app, fault)
                app_statistics[app + '-' + fault + '-' + str(device_number)] = app_stat.get_resource_summary_matrix()

    jitter_heatmap_data = []
    print("-------------------")
    print(heatmap_x_labels)
    print(heatmap_y_labels)
    print("-------------------")
    for app in all_apps:
        for fault in faults:
            faulty_cpu_usr = []
            faulty_cpu_sys = []
            faulty_mem = []
            faulty_disk = []
            faulty_net_tra = []
            faulty_net_rec = []

            fault_free_cpu_usr = []
            fault_free_cpu_sys = []
            fault_free_mem = []
            fault_free_disk = []
            fault_free_net_tra = []
            fault_free_net_rec = []

            for device_number in range(max_edge_devices):
                if device_number == 0 and edge_device_filename == "raspberrypi":
                    continue
                if edge_device_filename == 'nano' and device_number == 1 and app == 'OD-CPU' and fault == 'MEM-60%':
                    continue
                summary_matrix = app_statistics[app + '-' + fault + '-' + str(device_number)]
                for key in summary_matrix.keys():
                    value = summary_matrix[key]
                    cpu_usr_value = resource_to_value_list('CPU-USER', value)
                    cpu_sys_value = resource_to_value_list('CPU-SYS', value)
                    mem_value = resource_to_value_list('MEM', value)
                    disk_value = resource_to_value_list('DISK', value)
                    net_tra_value = resource_to_value_list('NET-TRA', value)
                    net_rec_value = resource_to_value_list('NET-REC', value)

                    if value.label != 'No-Fault':
                        if cpu_usr_value is not None:
                            faulty_cpu_usr.append(cpu_usr_value)
                        if cpu_sys_value is not None:
                            faulty_cpu_sys.append(cpu_sys_value)
                        if mem_value is not None:
                            faulty_mem.append(mem_value)
                        if disk_value is not None:
                            faulty_disk.append(disk_value)
                        if net_tra_value is not None:
                            faulty_net_tra.append(net_tra_value)
                        if net_rec_value is not None:
                            faulty_net_rec.append(net_rec_value)
                    else:
                        print(f'value label: {value.label}')
                        if cpu_usr_value is not None:
                            fault_free_cpu_usr.append(cpu_usr_value)
                        if cpu_sys_value is not None:
                            fault_free_cpu_sys.append(cpu_sys_value)
                        if mem_value is not None:
                            fault_free_mem.append(mem_value)
                        if disk_value is not None:
                            fault_free_disk.append(disk_value)
                        if net_tra_value is not None:
                            fault_free_net_tra.append(net_tra_value)
                        if net_rec_value is not None:
                            fault_free_net_rec.append(net_rec_value)

            print(f'{app} -- {fault}')

            if fault == 'No-Fault':
                # fault_free_cpu_usr = fault_free_cpu_usr[0: len(fault_free_cpu_usr)-15]
                # fault_free_cpu_sys = fault_free_cpu_sys[0: len(fault_free_cpu_sys)-15]
                # fault_free_mem = fault_free_mem[0: len(fault_free_mem)-15]
                # fault_free_disk = fault_free_disk[0: len(fault_free_disk)-15]
                jitter_heatmap_data.append([(app + '-CPU-USER'), fault, jitter(fault_free_cpu_usr)])
                jitter_heatmap_data.append([(app + '-CPU-SYS'), fault, jitter(fault_free_cpu_sys)])
                jitter_heatmap_data.append([(app + '-MEM'), fault, jitter(fault_free_mem)])
                jitter_heatmap_data.append([(app + '-DISK'), fault, jitter(fault_free_disk)])
            else:
                jitter_heatmap_data.append([(app + '-CPU-USER'), fault, jitter(faulty_cpu_usr)])
                jitter_heatmap_data.append([(app + '-CPU-SYS'), fault, jitter(faulty_cpu_sys)])
                jitter_heatmap_data.append([(app + '-MEM'), fault, jitter(faulty_mem)])
                jitter_heatmap_data.append([(app + '-DISK'), fault, jitter(faulty_disk)])

    draw_heatmap(heatmap_x_labels, heatmap_y_labels, jitter_heatmap_data, edge_device,
                 'Impact of Faults on Resource Utilization Jitter Heatmap on ' + edge_device)


#####################################################################################################
######################################################################################################
######################################################################################################
def compare_ML_accuracies_on_detection():
    device = 'raspberrypi'
    device2 = 'nano'
    ml_accuracies = pd.read_csv(f'MLResults-{device}-merged.csv', delimiter=',', skiprows=3)
    ml_accuracies2 = pd.read_csv(f'MLResults-{device2}-merged.csv', delimiter=',', skiprows=3)
    print(ml_accuracies['APP_FILE'].unique())
    print(ml_accuracies2['APP_FILE'].unique())
    # Define machine learning method columns
    ml_methods = ['TREE', 'NEARESTNEIGHBOR', 'FOREST', 'MLP']
    ml_methods_name = ['Tree', 'Nearest Neighbor', 'Forest', 'MLP']
    apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF', 'IP', 'SA', 'PS', 'AE', 'OD-CPU', 'IC-A-CPU']
    apps2 = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF', 'IP', 'SA', 'PS', 'AE', 'OD-CPU', 'OD-GPU']
    # if device == 'nano':


    # Prepare data for plotting
    fig, axes = plt.subplots(2, 2, figsize=(19, 10))

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.9, bottom=0.1, left=0.05, right=0.95)

    # Set the font family to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    # Set the CUD palette
    cud_palette = sns.color_palette("colorblind", n_colors=12)
    hatch_patterns = ['', '\\\\\\', 'xxx', '---', '////', '\\\\\\\\', '....',
                      '++++', '||||', 'xxxx', '////', '\\\\\\\\']

    app_styles = {}
    for (i, app) in zip(range(len(apps)), apps):
        app_styles[app] = {'color': cud_palette[i], 'hatch': hatch_patterns[i]}

    # Create an empty list to collect legend handles and labels
    legend_handles = []
    for idx, ml_method in enumerate(ml_methods):
        ml_acc_column = ml_method + '_ACC'

        # Extract data for the current ML method
        window_size = ml_accuracies['EARLY_PREDICTION_IN_SECONDS']
        app_names = ml_accuracies['APP_FILE']
        accuracy_values = ml_accuracies[ml_acc_column]

        app_names2 = ml_accuracies2['APP_FILE']
        window_size2 = ml_accuracies2['EARLY_PREDICTION_IN_SECONDS']
        accuracy_values2 = ml_accuracies2[ml_acc_column]

        # print(f"-------------{ml_method}")
        # print(app_names, accuracy_values)

        result_to_plot = {}
        for window_s, app, accuracy in zip(window_size, app_names, accuracy_values):
            if window_s == 0:
                app_name = app.replace(f'summary-{device}-', '').replace('.csv', '')
                result_to_plot[app_name+'-'+device] = accuracy
            else:
                break
        for window_s, app, accuracy in zip(window_size2, app_names2, accuracy_values2):
            if window_s == 0:
                app_name = app.replace(f'summary-{device2}-', '').replace('.csv', '')
                result_to_plot[app_name+'-'+device2] = accuracy
            else:
                break

        # Prepare data for plotting
        row_idx = idx // 2
        col_idx = idx % 2
        ax = axes[row_idx, col_idx]
        for i, app in enumerate(apps):
            style = app_styles[app]
            accuracy_value_1 = result_to_plot[app + '-' + device]
            if app == 'IC-A-CPU':
                accuracy_value_2 = result_to_plot['OD-GPU-' + device2]
            else:
                accuracy_value_2 = result_to_plot[app + '-' + device2]

            ax.bar(app+'-'+device, accuracy_value_1*100, color=style['color'], hatch=style['hatch'], label=app)
            if app == 'IC-A-CPU':
                ax.bar('OD-GPU-' + device2, accuracy_value_2*100, color=style['color'], hatch=style['hatch'], label=app)
            else:
                ax.bar(app+'-'+device2, accuracy_value_2*100, color=style['color'], hatch=style['hatch'], label=app)

        ax.set_title(f'Accuracies of {ml_methods_name[idx]} Method',fontsize=16,  fontweight='bold')
        # if device == 'nano':
        #     pass
        # else:
        #     ax.set_title(f'RPi: {ml_methods_name[idx]} Method', fontsize=16, fontweight='bold')

        ax.set_ylim(0, 100)
        # ax.set_xlabel('Application', fontsize=16, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
        ax.tick_params(axis='both', which='minor', labelsize=14, width=2, length=4)
        # Set the spine properties to make the subplot box border bold
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # Set border width
        x_ticks = ['FFT-RPi', 'FFT-JNano', 'FPO-SIN-RPi', 'FPO-SIN-JNano', 'FPO-SQRT-RPi', 'FPO-SQRT-JNano', 'SORT-RPi', 'SORT-JNano', 'DD-RPi', 'DD-Nano', 'IPERF-RPi', 'IPERF-JNano', 'IP-RPi', 'IP-JNano', 'SA-RPi', 'SA-JNano', 'PS-RPi', 'PS-JNano', 'AE-RPi', 'AE-JNano', 'OD-CPU-RPi', 'OD-CPU-JNano', 'IC-A-CPU-RPi', 'OD-GPU-JNano']
        ax.set_xticklabels(x_ticks, rotation=90)
        ax.grid(True)

    plt.subplots_adjust(top=0.95)
    # Rotate x-axis labels for all subplots

    plt.tight_layout()
    plt.show()


def draw_ML_accuracy_on_window_size():
    device = 'raspberrypi'
    # device = 'nano'
    ml_accuracies = pd.read_csv(f'MLResults-{device}-merged.csv', delimiter=',', skiprows=3)
    print(ml_accuracies['APP_FILE'].unique())
    # Define machine learning method columns
    ml_methods = ['TREE', 'SVM_RBF', 'SVM_SIG', 'NEARESTNEIGHBOR', 'FOREST', 'MLP']
    ml_methods_name = ['Tree', 'SVG RBF', 'SVM Sigmoid', 'Nearest Neighbor', 'Forest', 'MLP']
    apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
            'IP', 'SA',  'PS', 'AE', 'OD-CPU', 'IC-A-CPU']
    if device == 'nano':
        apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
                'IP', 'SA', 'PS', 'AE', 'OD-CPU', 'OD-GPU']

    # Prepare data for plotting
    window_sizes = list(range(11))
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.9, bottom=0.1, left=0.05, right=0.95)

    line_styles = ['-', '--', '-.', ':']  # Different line styles
    # colors = sns.color_palette("colorblind", n_colors=12)

    colors = plt.cm.get_cmap('tab20', len(apps))  # Get distinct colors based on unique apps
    # Define distinct line styles for each app
    distinct_line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-',
                            '--', '-.', ':']

    # Create an empty list to collect legend handles and labels
    legend_handles = []
    for idx, ml_method in enumerate(ml_methods):
        ml_acc_column = ml_method + '_ACC'

        # Extract data for the current ML method
        app_names = ml_accuracies['APP_FILE']
        # if ml_method == 'SVM_RBF' or ml_method == 'SVM_POLY':
        #     accuracy_values = ml_2_accuracies[ml_acc_column]
        # else:
        accuracy_values = ml_accuracies[ml_acc_column]

        result_to_plot = {}
        for app, accuracy in zip(app_names, accuracy_values):
            app_name = app.replace(f'summary-{device}-', '').replace('.csv', '')
            if app_name not in result_to_plot.keys():
                result_to_plot[app_name] = []
            result_to_plot[app_name].append([window_sizes, accuracy])

        # Prepare data for plotting
        row_idx = idx // 3
        col_idx = idx % 3
        ax = axes[row_idx, col_idx]

        for i, app in enumerate(apps):
            accuracy_values = result_to_plot[app]
            x_values, y_values = zip(*accuracy_values)
            line_style = line_styles[i % len(line_styles)]
            color = colors(i)
            line_style = distinct_line_styles[i % len(distinct_line_styles)]

            ax.plot(window_sizes, y_values, marker='o', linestyle=line_style, color=color, label=app)
            # ax.plot(window_sizes, y_values, marker='o', linestyle=line_style, color=color, label=app)


        ax.set_ylim(0.4, 1)
        ax.set_xlabel('Window Size', fontsize=16, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=16, fontweight='bold')

        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
        ax.tick_params(axis='both', which='minor', labelsize=14, width=2, length=4)
        if device == 'nano':
            ax.set_title(f'JNano: {ml_methods_name[idx]} Method',fontsize=16,  fontweight='bold')
        else:

            ax.set_title(f'RPi: {ml_methods_name[idx]} Method', fontsize=16, fontweight='bold')
        ax.set_xticks(window_sizes)
        # Set the spine properties to make the subplot box border bold
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # Set border width

        # Collect handles and labels for the legend
        handles, labels = ax.get_legend_handles_labels()
        legend_handles.extend(handles)
        # ax.legend()
        ax.grid(True)

    # Create a common legend outside the figure horizontally
    legend = fig.legend(legend_handles, labels, loc='upper center', fontsize=14, ncol=len(apps))

    # Make legend labels bold
    for text in legend.get_texts():
        text.set_fontweight('bold')

    # Adjust layout and show the plots
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ######################################################################
    ## Tested for creating summary files for machine learning code
    ######################################################################
    edge_device = 'Raspberry Pi'
    # edge_device = 'Jetson Nano'

    if edge_device == 'Raspberry Pi':
        all_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
                    'IP', 'SA', 'PS', 'AE', 'OD-CPU', 'IC-A-CPU']
        device_filename = 'raspberrypi'
    else:
        all_apps = ['FFT', 'FPO-SIN', 'FPO-SQRT', 'SORT', 'DD', 'IPERF',
                    'IP', 'SA', 'PS', 'AE', 'OD-CPU', 'OD-GPU']
        device_filename = 'nano'

    # for app in all_apps:
    #     app_statistics = AppStatistics(app, device_filename)

    ######################################################################
    # ********************************************************************#
    ######################################################################
    ## 1- Tested and it works for drawing fault free resource utilization for all apps (micro and applications)
    # draw_fault_free_resource_comparisons('MEM')
    ######################################################################
    # ********************************************************************#
    ######################################################################
    ## 2- Tested and it works for drawing resource utils of micros/macros with subfigures on each fault and No-Fault
    # draw_all_faults_resource_comparisons('MICRO', 'CPU-USR')
    ######################################################################
    # ********************************************************************#
    ######################################################################
    ## 3- Tested and it works for drawing resource utils of a SINGLE APP with subfigures on each fault and No-Fault
    # edge_device = 'Raspberry Pi'
    # edge_device = 'Jetson Nano'
    # draw_resource_comparisons_single_app_all_faults(edge_device, 'IP', 'CPU-SYS')
    # draw_single_app_all_resource_comp(edge_device, 'IP', 'PF-0')
    ######################################################################
    # ********************************************************************#
    ######################################################################
    ##******  4- Tested for drawing latency of SINGLE App under all faults with subfigures rtt, comp_t, transmit_t
    # draw_latency_single_app_all_faults(edge_device, 'FFT')
    # edge_device = 'Jetson Nano'
    # edge_device = 'Raspberry Pi'
    # draw_latency_single_app_over_time(edge_device, 'FFT')
    ######################################################################
    # ********************************************************************#
    ######################################################################
    ## 5- Tested drawing heatmap to find jitter on resource utils all faults all apps
    draw_heatmaps_on_resource_jitter(edge_device)
    ######################################################################
    # ********************************************************************#
    ######################################################################
    ## 6- Tested drawing heatmap to find change on average on resource utils all faults all apps
    # edge_device = 'Jetson Nano'
    # edge_device = 'Raspberry Pi'
    # draw_heatmaps_on_resource_increase(edge_device)
    # draw_heatmaps_on_resource_jitter(edge_device)

    ######################################################################
    # ********************************************************************#
    ######################################################################
    ## 7- Tested drawing heatmap to find change on average computation time all faults all apps
    ## 8- Tested drawing heatmap to find jitter on computation time all faults all apps
    # draw_heatmaps_on_latency_increase()
    # draw_heatmaps_on_latency_increase_maybe_better_version()
    # draw_heatmaps_on_latency_jitter()

    ######################################################################
    # ********************************************************************#
    ######################################################################
    ## 9- Comparing Machine Learning Methods Accuracy of Classification Based on Window Size
    # draw_ML_accuracy_on_window_size()
    # compare_ML_accuracies_on_detection()
    ######################################################################
    # ********************************************************************#
    ######################################################################
    ## 10 - Comparing all latencies all apps under faults
    # draw_latency_all_apps_all_faults(edge_device)

    # draw_latency_all_apps_subfigures(edge_device, 'micros', 'rtt')
    ######################################################################
    # ********************************************************************#
    ######################################################################
    ## 11 - Draw single app - single utilization - single fault
    # draw_single_utilization(edge_device, 'IP', 'IO-100', 'MEM')
    ######################################################################
    # ********************************************************************#
    ######################################################################
    # draw_faulty_resource_comparisons('MEM')

    # 1) Show fail-slow directly
    # draw_latency_single_app_over_time(edge_device, 'IC-A-CPU')
    #
    # # 2) Correlate with resource saturation in the same run/fault
    # draw_single_app_all_resource_comp(edge_device, 'IC-A-CPU', 'MEM-60%')
    #
    # # 3) Summary bar view for a single app
    # draw_latency_single_app_all_faults(edge_device, 'IC-A-CPU')
