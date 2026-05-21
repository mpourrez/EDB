import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math
import csv
import configs

class AppFaultStatistics:
    def __init__(self, device, app, fault):
        super().__init__()
        self.device = device
        self.app = app
        self.fault = fault

        # Fault Injection Times
        self.fault_start_times, self.fault_stop_times = None, None
        # CPU Utilizations
        self.cpu_datetimes, self.user_cpu_utilization, self.sys_cpu_utilization, self.total_cpu_utilization = None, None, None, None
        # CPU Temperatures
        self.cpu_temp_datetimes, self.cpu_temperatures = None, None
        # Latencies
        self.request_received_datetimes, self.response_datetimes, self.rtt, self.compute_times, self.transmission_times = None, None, None, None, None
        # Memory Utilizations
        self.mem_datetimes, self.mem_utilization = None, None
        # Network Utilizations
        self.net_datetimes, self.net_utilizations, self.net_receive_rates, self.net_transmit_rates = None, None, None, None
        # Disk Utilizations
        self.io_datetimes, self.disk_utilization = None, None
        self.experiment_duration = configs.EXPERIMENT_DURATION
        self.read_all_saved_statistics()

    def get_base_date(self):
        first_datetime_in_latency = self.request_received_datetimes[0]
        base_date = first_datetime_in_latency.strftime('%Y-%m-%d')
        return base_date + ' '
        # if self.device != 'nano':
        #     if self.fault == 'No-Fault' or self.app == 'PS' or self.app == 'AE' or self.app == 'OD-CPU':
        #         return '2023-06-01 '
        #     if self.app == 'IC-S-CPU' or (self.app == 'IC-A-CPU' and 'CPU' not in self.fault):
        #         return '2023-05-31 '
        #     if self.fault == 'PING' or self.fault == 'TCP':
        #         return '2023-08-15 '
        #     return '2023-05-30 '
        # else:
        #     if self.app == 'FFT':
        #         return '2023-08-12 '
        #     if self.app == 'FPO-SIN':
        #         return '2023-06-19 '
        #     if self.app == 'FPO-SQRT' and 'CPU' in self.fault:
        #         return '2023-06-19 '
        #     if self.app == 'FPO-SQRT' and self.fault == 'CCHE-0':
        #         return '2023-06-19 '
        #     if self.app == 'SORT' and (self.fault == 'MEM-60%' or self.fault == 'PF-0'):
        #         return '2023-08-12 '
        #     if self.app == 'SORT' and 'CPU' not in self.fault:
        #         return '2023-06-19 '
        #
        #     else:
        #         return '2023-06-20 '


    def read_all_saved_statistics(self):
        self.get_latencies()
        self.get_fault_injection_times()
        if len(self.fault_start_times) == 0:
            self.experiment_duration = configs.NUMBER_OF_FAULT_FREE_ROUNDS * configs.FAULT_FREE_DURATIONS
        self.get_cpu_utilization()
        self.get_cpu_temperatures()
        self.get_iostat()
        self.get_network_statistics()
        self.get_memory_utilization()

    def get_resource_datetimes(self, resource):
        if resource == 'CPU':
            resource_datetime = self.cpu_datetimes
        elif resource == 'MEM':
            resource_datetime = self.mem_datetimes
        elif resource == 'NET':
            resource_datetime = self.net_datetimes
        elif resource == 'DISK':
            resource_datetime = self.io_datetimes
        elif resource == 'LATENCY':
            resource_datetime = self.request_received_datetimes
        else:
            resource_datetime = self.cpu_temp_datetimes
        return resource_datetime

    def get_resource_values(self, resource, rtype):
        values = None
        if resource == 'CPU' and rtype == 'SYS':
            values = self.sys_cpu_utilization
        elif resource == 'CPU' and rtype == 'USR':
            values = self.user_cpu_utilization
        elif resource == 'CPU' and rtype == 'UTIL':
            values = self.total_cpu_utilization
        elif resource == 'MEM':
            values = self.mem_utilization
        elif resource == 'NET' and rtype == 'REC':
            values = self.net_receive_rates
        elif resource == 'NET' and rtype == 'TRA':
            values = self.net_transmit_rates
        elif resource == 'NET' and rtype == 'UTIL':
            values = self.net_utilizations
        elif resource == 'DISK':
            values = self.disk_utilization
        elif resource == 'LATENCY' and rtype == 'COM-TIME':
            values = self.compute_times
        elif resource == 'LATENCY' and rtype == 'TRA-TIME':
            values = self.transmission_times
        else:
            values = self.cpu_temperatures
        return values

    def get_resource_metric(self, resource, rtype):
        if resource == 'NET' and rtype == 'REC':
            metric = 'kB/s'
        elif resource == 'NET' and rtype == 'TRA':
            metric = 'kB/s'
        elif resource == 'LATENCY' and rtype == 'COM-TIME':
            metric = 'ms'
        elif resource == 'LATENCY' and rtype == 'TRA-TIME':
            metric = 'ms'
        else:
            metric = '%'
        return metric

    def get_faulty_values(self, resource, rtype):
        resource_datetime = self.get_resource_datetimes(resource)
        resource_values = self.get_resource_values(resource, rtype)
        fault_start_indices, fault_end_indices = extract_fault_start_end_indices(resource_datetime,
                                                                                 self.fault_start_times)
        _, resource_faulty_values = get_fault_free_faulty_values(resource_values, fault_start_indices,
                                                                 fault_end_indices)
        return resource_faulty_values

    def get_fault_free_values(self, resource, rtype):
        resource_datetime = self.get_resource_datetimes(resource)
        resource_values = self.get_resource_values(resource, rtype)
        fault_start_indices, fault_end_indices = extract_fault_start_end_indices(resource_datetime,
                                                                                 self.fault_start_times)
        resource_fault_free_values, _ = get_fault_free_faulty_values(resource_values, fault_start_indices,
                                                                     fault_end_indices)
        return resource_fault_free_values

    def get_fault_injection_times(self):
        fault_injection_times = pd.read_csv(
            'results_over_time_old/'+self.device+'/' + self.app + '-' + self.fault + '-FaultInjection.csv',
            # 'results_over_time_old/Python3.9Results/'+self.device+'/' + self.app + '-' + self.fault + '-FaultInjection.csv',
            delimiter=',')

        fault_start_times_cl = fault_injection_times['fault_injection_start_time']
        fault_stop_times_cl = fault_injection_times['fault_injection_stop_time']
        self.fault_start_times = []
        self.fault_stop_times = []

        # for start_t, stop_t in zip(fault_start_times_cl, fault_stop_times_cl):
        #     fault_start_time = get_datetime_from_timestamp(self.app, self.fault, start_t)
        #     fault_end_time = get_datetime_from_timestamp(self.app, self.fault, stop_t)
        #     self.fault_start_times.append(fault_start_time)
        #     self.fault_stop_times.append(fault_end_time)

        for start_t in fault_start_times_cl:
            fault_start_time = get_datetime_from_timestamp(self.app, self.fault, start_t)
            self.fault_start_times.append(fault_start_time)

    #############################################
    #########     CPU Utilization    ############
    #############################################
    def get_cpu_utilization(self):
        date = self.get_base_date()
        # cpu_df = pd.read_csv('results_over_time_old/Python3.9Results/'+self.device+'/' + self.app + '-' + self.fault + '-CPU.txt',
        cpu_df = pd.read_csv('results_over_time_old/'+self.device+'/' + self.app + '-' + self.fault + '-CPU.txt',
                             delimiter='\s+', skiprows=2)
        cpu_times = cpu_df.iloc[:, 0]
        cpu_am_pm = cpu_df.iloc[:, 1]
        self.cpu_datetimes = []
        previous_datetime = None
        for (t, am_pm) in zip(cpu_times, cpu_am_pm):
            if 'Average' not in t:
                cpu_datetime = datetime.datetime.strptime(date + t + ' ' + am_pm, '%Y-%m-%d %I:%M:%S %p')
                if previous_datetime is not None and previous_datetime.hour == 23 and cpu_datetime.hour == 0:
                    cpu_datetime += datetime.timedelta(days=1)
                else:
                    previous_datetime = cpu_datetime

                self.cpu_datetimes.append(cpu_datetime)
        # self.cpu_datetimes = self.cpu_datetimes[:self.experiment_duration + 1]
        self.user_cpu_utilization = cpu_df['%usr'] #[:self.experiment_duration + 1]
        self.sys_cpu_utilization = cpu_df['%sys'] #[:self.experiment_duration + 1]
        self.total_cpu_utilization = 100 - cpu_df['%idle'] #[:self.experiment_duration + 1]
        # print(self.cpu_datetimes)
        # print()
        # print()

    def get_experiment_start_datetime(self):
        return self.cpu_datetimes[0]
    #############################################
    #########     CPU Temperatures    ###########
    #############################################
    def get_cpu_temperatures(self):
        # cpu_temp_df = pd.read_csv('results_over_time_old/Python3.9Results/'+self.device+'/' + self.app + '-' + self.fault + '-TEMP.txt',
        cpu_temp_df = pd.read_csv('results_over_time_old/'+self.device+'/' + self.app + '-' + self.fault + '-TEMP.txt',
                                  delimiter=',')
        cpu_temp_times = cpu_temp_df['Timestamp_ms']
        self.cpu_temp_datetimes = []

        first_date = self.cpu_datetimes[0]
        first_index = 0
        last_date = self.cpu_datetimes[len(self.cpu_datetimes) - 1]
        last_index = 0
        for t in cpu_temp_times:
            cpu_temp_datetime = get_datetime_from_timestamp(self.app, self.fault, t)
            cpu_temp_datetime = cpu_temp_datetime.replace(microsecond=0)
            if cpu_temp_datetime < first_date:
                first_index += 1
                continue
            if cpu_temp_datetime > last_date:
                break
            self.cpu_temp_datetimes.append(cpu_temp_datetime)
            last_index += 1
        self.cpu_temperatures = cpu_temp_df['CPU_Temp'][first_index:last_index + 1]
        # print("checking cpu temperature timestamps")
        # print(self.cpu_temp_datetimes[0])
        # print(self.cpu_temp_datetimes[len(self.cpu_temp_datetimes) - 1])
        # print(self.cpu_temp_datetimes)


    #############################################
    #########     Latencies    ##################
    #############################################
    def get_latencies(self):
        # latencies_df = pd.read_csv('results_over_time_old/Python3.9Results/'+self.device+'/' + self.app + '-' + self.fault + '-Latency.csv',
        latencies_df = pd.read_csv('results_over_time_old/'+self.device+'/' + self.app + '-' + self.fault + '-Latency.csv',
                                   delimiter=',')
        request_times = latencies_df['request_received_time_ms']
        response_times = latencies_df['response_time_ms']
        self.compute_times = latencies_df['compute_time']
        self.transmission_times = latencies_df['transmission_time']
        self.rtt = latencies_df['end_to_end_latency']

        self.request_received_datetimes = []
        self.response_datetimes = []

        for req, res in zip(request_times, response_times):
            self.request_received_datetimes.append(get_datetime_from_timestamp(self.app, self.fault, req))
            self.response_datetimes.append(get_datetime_from_timestamp(self.app, self.fault, res))

        print("checking latency timestamps")
        print(self.request_received_datetimes[0])
        print(self.request_received_datetimes[len(self.request_received_datetimes)-1])
        # print(self.response_datetimes)
        # print(len(self.response_datetimes))

    #############################################
    #########     Disk Utilization    ###########
    #############################################
    def get_iostat(self):
        date = self.get_base_date()
        # io_df = pd.read_csv('results_over_time_old/Python3.9Results/'+self.device+'/' + self.app + '-' + self.fault + '-IO.txt', delimiter='\s+',
        io_df = pd.read_csv('results_over_time_old/'+self.device+'/' + self.app + '-' + self.fault + '-IO.txt', delimiter='\s+',
                            skiprows=3)
        self.io_datetimes = []
        self.disk_utilization = []  # %util
        writes_ps = []  # w/s
        write_kilobytes = []  # wkB/s
        write_wait_times = []  # w_await
        first = True
        first_date = self.cpu_datetimes[0]
        last_date = self.cpu_datetimes[len(self.cpu_datetimes) - 1]
        print('First Global Date {0}'.format(first_date))
        print('Last Global Date {0}'.format(last_date))
        previous_datetime = None
        for i in range(len(io_df)):
            if '/2023' in io_df.iloc[:, 0][i]:
                io_time = io_df.iloc[:, 1][i]
                am_pm = io_df.iloc[:, 2][i]
                io_datetime = datetime.datetime.strptime(date + io_time + ' ' + am_pm, '%Y-%m-%d %I:%M:%S %p')
                io_datetime = io_datetime.replace(microsecond=0)
                if previous_datetime is not None and previous_datetime.hour == 23 and io_datetime.hour == 0:
                    io_datetime += datetime.timedelta(days=1)
                else:
                    previous_datetime = io_datetime
                if io_datetime < first_date:
                    continue
                if io_datetime > last_date:
                    break
                if first:
                    new_second = io_datetime.second - 1
                    if new_second < 0:
                        new_second = 59
                    first_missing_date = io_datetime.replace(second=new_second)
                    if first_missing_date < first_date:
                        continue
                    self.io_datetimes.append(first_missing_date)
                    first = False
                self.io_datetimes.append(io_datetime)
            elif io_df.iloc[:, 0][i] == 'mmcblk0':
                if 'nano' in self.device:
                    self.disk_utilization.append(float(io_df.iloc[:, 15][i]))
                else:
                    self.disk_utilization.append(float(io_df.iloc[:, 22][i]))
        # print(self.io_datetimes)

    #############################################
    #########     NEtwork Utilizations    #######
    #############################################
    def get_network_statistics(self):
        date = self.get_base_date()

        # net_df = pd.read_csv('results_over_time_old/Python3.9Results/'+self.device+'/' + self.app + '-' + self.fault + '-NET.txt',
        net_df = pd.read_csv('results_over_time_old/'+self.device+'/' + self.app + '-' + self.fault + '-NET.txt',
                             delimiter='\s+',
                             skiprows=2)
        self.net_datetimes = []
        self.net_utilizations = []  # %ifutil
        self.net_receive_rates = []  # rxkB/s The number of kilobytes received per second.
        self.net_transmit_rates = []  # txkB/s: The number of kilobytes transmitted per second

        last_date = self.cpu_datetimes[len(self.cpu_datetimes) - 1]
        previous_datetime = None
        for i in range(len(net_df)):
            if net_df.iloc[:, 2][i] == 'wlan0':
                net_time = net_df.iloc[:, 0][i]
                net_am_pm = net_df.iloc[:, 1][i]
                net_datetime = datetime.datetime.strptime(date + net_time + ' ' + net_am_pm, '%Y-%m-%d %I:%M:%S %p')
                if previous_datetime is not None and previous_datetime.hour == 23 and net_datetime.hour == 0:
                    net_datetime += datetime.timedelta(days=1)
                else:
                    previous_datetime = net_datetime
                if net_datetime > last_date:
                    break
                self.net_datetimes.append(net_datetime)

                self.net_utilizations.append(float(net_df.iloc[:, 10][i]))
                self.net_receive_rates.append(float(net_df.iloc[:, 5][i]))
                self.net_transmit_rates.append(float(net_df.iloc[:, 6][i]))

        # if self.app=='IPERF':
        #     index_of_zero = 0
        #     for i, val in enumerate(self.net_receive_rates[1:], start=1):
        #         if all(x == 0.0 for x in self.net_receive_rates[i:]):
        #             index_of_zero = i
        #             break
        #     # index_of_zero = net_float_receive_rates.index(19291.64)
        #     print(index_of_zero)
        #     self.net_utilizations = self.net_utilizations[:index_of_zero]
        #     self.net_receive_rates = self.net_receive_rates[:index_of_zero]
        #     self.net_transmit_rates = self.net_transmit_rates[:index_of_zero]
        #     self.net_datetimes = self.net_datetimes[:index_of_zero]

    #############################################
    #########     Memory Utilizations    ########
    #############################################
    def get_memory_utilization(self):
        date = self.get_base_date()
        # mem_df = pd.read_csv('results_over_time_old/Python3.9Results/'+self.device+'/' + self.app + '-' + self.fault + '-MEM.txt',
        mem_df = pd.read_csv('results_over_time_old/'+self.device+'/' + self.app + '-' + self.fault + '-MEM.txt',
                             delimiter='\s+', skiprows=2)
        mem_times = mem_df.iloc[:, 0]
        mem_am_pm = mem_df.iloc[:, 1]
        self.mem_datetimes = []
        previous_datetime = None
        for (t, am_pm) in zip(mem_times, mem_am_pm):
            if 'Average' not in t:
                mem_datetime = datetime.datetime.strptime(date + t + ' ' + am_pm, '%Y-%m-%d %I:%M:%S %p')
                if previous_datetime is not None and previous_datetime.hour == 23 and mem_datetime.hour == 0:
                    mem_datetime += datetime.timedelta(days=1)
                else:
                    previous_datetime = mem_datetime
                self.mem_datetimes.append(mem_datetime)

        self.mem_utilization = mem_df['%memused']#[:self.experiment_duration + 1]
        # self.mem_datetimes = self.mem_datetimes[:self.experiment_duration + 1]

    ######################################################################################################################
    #################################     END: Reading Files      ########################################################
    ######################################################################################################################

    def plot_resource(self, resource, rtype, show_injection_times):
        resource_datetimes = self.get_resource_datetimes(resource)
        resource_values = self.get_resource_values(resource, rtype)
        resource_metric = self.get_resource_metric(resource, rtype)
        fault_start_indices, fault_end_indices = extract_fault_start_end_indices(resource_datetimes,
                                                                                 self.fault_start_times)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
        axs.plot(resource_datetimes, resource_values, label='memused')
        if show_injection_times:
            for start in fault_start_indices:
                axs.axvline(x=resource_datetimes[start], color='r')
            for end in fault_end_indices:
                axs.axvline(x=resource_datetimes[end], color='b')
        axs.set_xlabel('Time', fontsize=16, fontweight='bold')
        axs.set_ylabel(resource + '-' + rtype + ' (' + resource_metric + ')', fontsize=16, fontweight='bold')
        # axs.set_xlim(right=230)
        # axs.set_xlim(left=-10)
        axs.grid(which="major")
        fig.suptitle(resource + '-' + rtype + ' Utilization Plot for App: ' + self.app + ' - Fault: ' + self.fault,
                     fontsize=16, fontweight='bold')
        plt.show()

    def jitter_faulty_fault_free(self, resource, rtype, is_faulty):
        resource_datetimes = self.get_resource_datetimes(resource)
        resource_values = self.get_resource_values(resource, rtype)
        fault_start_indices, fault_end_indices = extract_fault_start_end_indices(resource_datetimes,
                                                                                 self.fault_start_times)
        faulty_jitters = []
        fault_free_jitters = []
        start_ind = 0
        for fault_start, fault_end in zip(fault_start_indices, fault_end_indices):
            fault_free_values = resource_values[start_ind:fault_start]
            faulty_values = resource_values[fault_start:fault_end]

            faulty_jitters.append(jitter(faulty_values))
            print(jitter(faulty_values))
            fault_free_jitters.append(jitter(fault_free_values))
            start_ind = fault_end + 1

        fault_free_values = resource_values[start_ind:len(resource_datetimes) - 85]
        fault_free_jitters.append(jitter(fault_free_values))

        average_fault_free_jitter = sum(fault_free_jitters) / len(fault_free_jitters)
        average_faulty_jitter = sum(faulty_jitters) / len(faulty_jitters)

        if is_faulty:
            print("faulty avg: {0}".format(average_faulty_jitter))
            return average_faulty_jitter
        print("fault free avg: {0}".format(average_fault_free_jitter))
        return average_fault_free_jitter

    def plot_resource(self, resource, rtype, show_injection_times):
        resource_datetimes = self.get_resource_datetimes(resource)
        resource_values = self.get_resource_values(resource, rtype)
        resource_metric = self.get_resource_metric(resource, rtype)
        fault_start_indices, fault_end_indices = extract_fault_start_end_indices(resource_datetimes,
                                                                                 self.fault_start_times)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
        axs.plot(resource_datetimes, resource_values, label='memused')
        if show_injection_times:
            for start in fault_start_indices:
                axs.axvline(x=resource_datetimes[start], color='r')
            for end in fault_end_indices:
                axs.axvline(x=resource_datetimes[end], color='b')
        axs.set_xlabel('Time', fontsize=16, fontweight='bold')
        axs.set_ylabel(resource + '-' + rtype + ' (' + resource_metric + ')', fontsize=16, fontweight='bold')
        # axs.set_xlim(right=230)
        # axs.set_xlim(left=-10)
        axs.grid(which="major")
        fig.suptitle(resource + '-' + rtype + ' Utilization Plot for App: ' + self.app + ' - Fault: ' + self.fault,
                     fontsize=16, fontweight='bold')
        plt.show()

    def plot_resource_with_residual(self, resource, rtype, show_injection_times):
        resource_datetimes = self.get_resource_datetimes(resource)
        resource_values = self.get_resource_values(resource, rtype)
        resource_metric = self.get_resource_metric(resource, rtype)
        fault_start_indices, fault_end_indices = extract_fault_start_end_indices(resource_datetimes,
                                                                                 self.fault_start_times)
        print(self.fault_start_times)
        print(resource_datetimes)

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
        axs.plot(resource_datetimes, resource_values, label=resource + '-' + rtype)

        residual_means = []
        residual_jitters = []
        residual_dates = []
        if show_injection_times:
            start_ind = 0
            for fault_start, fault_end in zip(fault_start_indices, fault_end_indices):
                axs.axvline(x=resource_datetimes[fault_start], color='r')
                axs.axvline(x=resource_datetimes[fault_end], color='b')

                fault_free_middle_date = resource_datetimes[start_ind] + (
                            resource_datetimes[fault_start] - resource_datetimes[start_ind]) / 2
                faulty_middle_date = resource_datetimes[fault_start] + (
                            resource_datetimes[fault_end] - resource_datetimes[fault_start]) / 2
                fault_free_values = resource_values[start_ind:fault_start]
                faulty_values = resource_values[fault_start:fault_end]

                residual_dates.append(fault_free_middle_date)
                residual_dates.append(faulty_middle_date)
                residual_means.append(sum(fault_free_values) / len(fault_free_values))
                residual_means.append(sum(faulty_values) / len(faulty_values))
                residual_jitters.append(jitter(fault_free_values))
                residual_jitters.append(jitter(faulty_values))
                start_ind = fault_end + 1

            extra_end = 85
            if resource == 'NET':
                extra_end = 1

            fault_free_middle_date = resource_datetimes[len(resource_datetimes) - extra_end] + (
                    resource_datetimes[len(resource_datetimes) - extra_end] - resource_datetimes[start_ind]) / 2
            fault_free_values = resource_values[start_ind:len(resource_datetimes) - extra_end]
            residual_dates.append(fault_free_middle_date)
            residual_means.append(sum(fault_free_values) / len(fault_free_values))
            residual_jitters.append(jitter(fault_free_values))

        axs.plot(residual_dates, residual_means, label='residual-mean')
        axs.plot(residual_dates, residual_jitters, label='residual-jitter')
        axs.set_xlabel('Time', fontsize=16, fontweight='bold')
        axs.set_ylabel(resource + '-' + rtype + ' (' + resource_metric + ')', fontsize=16, fontweight='bold')
        # axs.legends()
        # axs.set_xlim(right=230)
        # axs.set_xlim(left=-10)
        axs.grid(which="major")
        fig.suptitle(resource + '-' + rtype + ' Utilization Plot for App: ' + self.app + ' - Fault: ' + self.fault,
                     fontsize=16, fontweight='bold')
        plt.show()

    def get_all_latencies(self):
        return self.rtt, self.compute_times, self.transmission_times

    def get_all_latencies_and_datetimes(self):
        return self.request_received_datetimes, self.response_datetimes, self.rtt, self.compute_times, self.transmission_times

    def get_all_fault_injection_start_stop_times(self):
        fault_end_datetimes = [dt + datetime.timedelta(seconds=configs.FAULT_INJECTION_DURATION)
                               for dt in self.fault_start_times]
        return self.fault_start_times, fault_end_datetimes

    def get_resource_summary_matrix(self):
        summary_matrix = {}

        for (cpu_date, cpu_user, cpu_sys, cpu_total) in \
                zip(self.cpu_datetimes, self.user_cpu_utilization, self.sys_cpu_utilization,
                    self.total_cpu_utilization):
            result_summary = ResultSummary(cpu_user, cpu_sys, cpu_total, 'No-Fault')
            summary_matrix[cpu_date] = result_summary

        for (cpu_temp_date, cpu_temp) in zip(self.cpu_temp_datetimes, self.cpu_temperatures):
            if cpu_temp_date in summary_matrix.keys():
                summary_matrix[cpu_temp_date].cpu_temp = cpu_temp

        for (mem_date, mem) in zip(self.mem_datetimes, self.mem_utilization):
            if mem_date in summary_matrix.keys():
                summary_matrix[mem_date].mem = mem

        for (disk_date, disk) in zip(self.io_datetimes, self.disk_utilization):
            if disk_date in summary_matrix.keys():
                summary_matrix[disk_date].disk = disk

        for (net_date, net_rec, net_tra) in zip(self.net_datetimes, self.net_receive_rates, self.net_transmit_rates):
            if net_date in summary_matrix.keys():
                summary_matrix[net_date].net_rec = net_rec
                summary_matrix[net_date].net_tra = net_tra

        for (request_received_datetime, rtt, compute_time, transmit_time) in \
                zip(self.request_received_datetimes, self.rtt, self.compute_times, self.transmission_times):
            if request_received_datetime.replace(microsecond=0) in summary_matrix.keys():
                summary_matrix[request_received_datetime.replace(microsecond=0)].com_t = compute_time
                summary_matrix[request_received_datetime.replace(microsecond=0)].tra_t = transmit_time
                summary_matrix[request_received_datetime.replace(microsecond=0)].rtt = rtt

        # Update fault labels according to injection times
        fault_end_datetimes = [dt + datetime.timedelta(seconds=configs.FAULT_INJECTION_DURATION)
                               for dt in self.fault_start_times]
        # print(fault_start)
        # print(self.fault_start_times)
        # print(fault_end_datetimes)
        # print()
        for (start_injection, end_injection) in zip(self.fault_start_times, fault_end_datetimes):
            # print('{0} - {1}'.format(start_injection, end_injection))
            # print("---------------------------------------------------------")
            for key in summary_matrix.keys():
                # print(key)
                if start_injection <= key <= end_injection:
                    summary_matrix[key].label = self.fault
                    # print("hereeeeeee")
                    # print(self.fault)
            # print('-------------------------------')


        return summary_matrix

        # for key in summary_matrix.keys():
        #     print('{0} -----  {1}'.format(key, summary_matrix[key]))


class ResultSummary:
    def __init__(self, cpu_user, cpu_sys, cpu_tot, label):
        super().__init__()
        self.cpu_user = cpu_user
        self.cpu_sys = cpu_sys
        self.cpu_tot = cpu_tot
        self.cpu_temp = '-----'
        self.mem = None
        self.disk = None
        self.net_rec = None
        self.net_tra = None
        self.com_t = -1
        self.tra_t = -1
        self.rtt = -1
        self.label = label

    def __str__(self):
        if self.cpu_temp == '-----':
            return 'cpu_usr: {0:.2f} - cpu_sys: {1:.2f} - cpu_tot: {2:.2f} - cpu_temp:{3} - mem: {4:.2f} - ' \
                   'disk: {5:.2f} - net_rec: {6:.2f} - net_tra: {7:.2f} - compute_time: {8}, ' \
                   'transmit_time:{9} - label: {10}'.format(
                self.cpu_user, self.cpu_sys, self.cpu_tot, self.cpu_temp, self.mem, self.disk, self.net_rec,
                self.net_tra, self.com_t, self.tra_t, self.label)

        return 'cpu_usr: {0:.2f} - cpu_sys: {1:.2f} - cpu_tot: {2:.2f} - cpu_temp:{3:.2f} - mem: {4:.2f} - ' \
               'disk: {5:.2f} - net_rec: {6:.2f} - net_tra: {7:.2f} - compute_time: {8}, ' \
                   'transmit_time:{9} - label: {10}'.format(
                self.cpu_user, self.cpu_sys, self.cpu_tot, self.cpu_temp, self.mem, self.disk, self.net_rec,
                self.net_tra, self.com_t, self.tra_t, self.label)


class AppStatistics:
    def __init__(self, app, device):
        super().__init__()
        self.app = app
        self.device = device
        self.combine_statistics()

    def combine_statistics(self):
        max_device_count = 3
        # with open('results_over_time_old/Python3.9Results/summary-'+self.device+'-'+self.app+'.csv', 'w', newline='') as file:
        with open('results_over_time_old/summary-'+self.device+'-'+self.app+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            headers = ['Time', 'CPU_USER', 'CPU_SYS', 'CPU_TOT', 'CPU_TEMP', 'MEM', 'DISK', 'NET_REC', 'NET_TRA',
                       'COM_T', 'TRA_T', 'Label']
            lable_dict = {'No-Fault': 0, 'CPU-20': 1, 'CPU-60': 2, 'CPU-90': 3, 'MEM-20%': 4, 'MEM-60%': 5,
                          'MEM-90%': 6, 'IO-100': 7, 'PF-0': 8, 'CCHE-0': 9}
            writer.writerow(headers)

            # Write No Fault Results
            for device_number in range(max_device_count):
                app_no_fault_statistics = AppFaultStatistics(self.device + str(device_number + 1), self.app, 'No-Fault')
                print(f"Finished calling - app fault statistic no fault {self.device} {device_number} - {self.app}")
                summary_matrix = app_no_fault_statistics.get_resource_summary_matrix()
                for key in summary_matrix.keys():
                    value = summary_matrix[key]
                    row = [key, value.cpu_user, value.cpu_sys,  ("%.2f" % value.cpu_tot)]
                    rest_row = [value.com_t, value.tra_t]
                    if value.cpu_temp != '-----':
                        row.append("%.2f" % value.cpu_temp)
                    else:
                        row.append(-1)
                    if value.mem is None:
                        row.append(-1)
                    else:
                        row.append(value.mem)
                    if value.disk is None:
                        row.append(-1)
                    else:
                        row.append(value.disk)
                    if value.net_rec is None:
                        row.append(-1)
                    else:
                        row.append(value.net_rec)
                    if value.net_tra is None:
                        row.append(-1)
                    else:
                        row.append(value.net_tra)
                    row.extend(rest_row)
                    row.extend([lable_dict[value.label]])
                    writer.writerow(row)

            faults = ['No-Fault', 'CPU-20', 'CPU-60', 'CPU-90', 'MEM-20%', 'MEM-60%', 'MEM-90%', 'IO-100', 'PF-0',
                      'CCHE-0']
            if self.device == 'nano':
                faults = ['No-Fault', 'CPU-20', 'CPU-60', 'CPU-90', 'MEM-20%', 'MEM-60%', 'IO-100', 'PF-0', 'CCHE-0']

            # Write Faulty Results
            for fault in faults:
                for device_number in range(max_device_count):
                    if self.device == 'nano' and device_number == 1 and self.app == 'OD-CPU' and fault == 'MEM-60%':
                        continue
                    print(f'{self.device} {device_number} {self.app} -- {fault}')
                    app_fault_statistics = AppFaultStatistics(self.device + str(device_number + 1), self.app, fault)
                    summary_matrix = app_fault_statistics.get_resource_summary_matrix()
                    for key in summary_matrix.keys():
                        value = summary_matrix[key]
                        row = [key, value.cpu_user, value.cpu_sys, ("%.2f" % value.cpu_tot)]
                        rest_row = [value.com_t, value.tra_t]
                        if value.cpu_temp != '-----':
                            row.append("%.2f" % value.cpu_temp)
                        else:
                            row.append(-1)
                        if value.mem is None:
                            row.append(-1)
                        else:
                            row.append(value.mem)
                        if value.disk is None:
                            row.append(-1)
                        else:
                            row.append(value.disk)
                        if value.net_rec is None:
                            row.append(-1)
                        else:
                            row.append(value.net_rec)
                        if value.net_tra is None:
                            row.append(-1)
                        else:
                            row.append(value.net_tra)
                        row.extend(rest_row)
                        row.extend([lable_dict[value.label]])
                        writer.writerow(row)
                #     break
                # break




def extract_fault_start_end_indices(resource_datetimes, fault_start_times):
    resource_fault_start_indices = find_fault_injection_indices_on_utilizations(resource_datetimes, fault_start_times)
    resource_fault_end_datetimes = [dt + datetime.timedelta(seconds=30) for dt in fault_start_times]
    resource_fault_end_indices = find_fault_injection_indices_on_utilizations(resource_datetimes,
                                                                              resource_fault_end_datetimes)

    return resource_fault_start_indices, resource_fault_end_indices

def find_fault_injection_indices_on_utilizations(utilization_times, fault_injection_times):
    index = 0
    fault_index = 0

    fault_injection_indices = []
    while index < len(utilization_times):
        if fault_index >= len(fault_injection_times):
            break
        if fault_injection_times[fault_index] < utilization_times[index]:
            fault_injection_indices.append(index)
            fault_index += 1
        index += 1

    return fault_injection_indices


def get_fault_free_faulty_values(values, fault_start_indices, fault_end_indices):
    index = 0
    fault_free_values = []
    faulty_values = []
    for i in range(len(values)):
        if index >= len(fault_start_indices) or i < fault_start_indices[index] and i < fault_end_indices[index]:
            fault_free_values.append(values[i])
        elif i >= fault_start_indices[index] and i >= fault_end_indices[index] and index < len(fault_start_indices):
            index += 1
            faulty_values.append(values[i])
        else:
            faulty_values.append(values[i])
    return fault_free_values, faulty_values

def get_datetime_from_timestamp(app, fault, timestamp):
    fault_start_time = datetime.datetime.fromtimestamp(timestamp / 1000)
    new_hour = fault_start_time.hour - 3
    if new_hour < 0:
        new_hour = 24 + new_hour
        fault_start_time -= datetime.timedelta(days=1)

    fault_start_time = fault_start_time.replace(hour=new_hour)
    # print(fault_start_time)
    # print(fault_start_time)
    # print("-------")
    return fault_start_time

def jitter(values):
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / float(n)
    variance = sum((x - mean) ** 2 for x in values) / float(n - 1)
    return math.sqrt(variance)