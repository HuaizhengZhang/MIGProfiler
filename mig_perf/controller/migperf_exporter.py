import os
from pathlib import Path

import pandas as pd
DATA_DIR = Path(os.getcwd()).parent.parent / Path("data/results/A30/")
METRICS_DIR = Path(os.getcwd()).parent.parent / Path("mig_perf/metrics/metrics.txt")


class MIGPerfExporter:
    def __init__(self):
        self.power_prometheus = [
            "# HELP mig_service_graphics_power_consumption_watt 99% to tail latency for one input sample in seconds.\n",
            "# TYPE mig_service_graphics_power_consumption_watt Untype\n"]
        self.fbusd_prometheus = [
            "# HELP mig_service_frame_buffer_used_MB 99% to tail latency for one input sample in seconds.\n",
            "# TYPE mig_service_frame_buffer_used_MB Untype\n"]
        self.gract_prometheus = [
            "# HELP mig_service_graphics_engine_activity 99% to tail latency for one input sample in seconds.\n",
            "# TYPE mig_service_graphics_engine_activity Untype\n"]
        self.throughput_prometheus = [
            "# HELP mig_service_throughput_per_second 99% to tail latency for one input sample in seconds.\n",
            "# TYPE mig_service_throughput_per_second Untype\n"]
        self.latency_prometheus = [
            "# HELP mig_service_latency_seconds 99% to tail latency for one input sample in seconds.\n",
            "# TYPE mig_service_latency_seconds Untype\n"]

    def query(self, model_name: str, workload: str):
        self.__init__()
        file_name = f"{model_name}_{workload}.csv"
        file_path = DATA_DIR / file_name
        metric_csv_format = pd.read_csv(file_path)
        if "seq_length" not in metric_csv_format.columns:
            for index, record in metric_csv_format.iterrows():
                metrics = self.format_transform(record, workload)
                self.latency_prometheus.append(metrics['latency'])
                self.throughput_prometheus.append(metrics['throughput'])
                self.gract_prometheus.append(metrics['gract'])
                self.fbusd_prometheus.append(metrics['fbusd'])
                self.power_prometheus.append(metrics['power'])

    def export(self):
        result_for_prometheus = self.latency_prometheus + self.throughput_prometheus + self.fbusd_prometheus + \
                                self.gract_prometheus + self.power_prometheus
        with open(METRICS_DIR, mode='w', encoding='utf-8') as f:
            f.writelines(result_for_prometheus)

    @staticmethod
    def format_transform(record: pd.Series, workload):
        model_name = str(record['model_name'])
        workload = str(workload)
        batch_size = str(record['batch_size'])
        mig_profile = str(record['mig_profile'])
        labels = "{" + \
                 f"model_name=\"{model_name}\",workload=\"{workload}\", mig_profile=\"{mig_profile}\",batch_size=\"{batch_size}\"" \
                 + "}"
        ret = {
            'latency': f"mig_service_latency_seconds{labels} {str(round(record['latency'] / 1000, 3))}\n",
            'throughput': f"mig_service_throughput_per_second{labels} {str(record['throughput'])}\n",
            'gract': f"mig_service_graphics_engine_activity{labels} {str(record['gract'])}\n",
            'fbusd': f"mig_service_frame_buffer_used_MB{labels} {str(record['fbusd'])}\n",
            'power': f"mig_service_graphics_power_consumption_watt{labels} {str(record['power'])}\n"
        }
        return ret


if __name__ == '__main__':
    exporter = MIGPerfExporter()
    exporter.query('vision_transformer', 'cv_infer')
    exporter.export()
