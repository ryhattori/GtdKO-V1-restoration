from __future__ import annotations

import importlib.util
import os


def _load_summary_module():
	module_path = os.path.join(os.path.dirname(__file__), 'group_analysis.py')
	spec = importlib.util.spec_from_file_location('summary_plots', module_path)
	module = importlib.util.module_from_spec(spec)
	assert spec.loader is not None
	spec.loader.exec_module(module)
	return module


def main() -> None:
	m = _load_summary_module()
	m.SHUFFLED_DECODING_NULL_MAX_WORKERS = len(m.GROUPS.get('RESCUED_BEFORE', []))

	print('[target] orientation_decoding_accuracy_by_sf_groups.svg')
	m.plot_metric(
		metric_key='decoding_accuracy',
		title='Continuous orientation decoding correlation vs spatial frequency',
		ylabel='Circular correlation',
		filename='orientation_decoding_accuracy_by_sf_groups.svg',
		ylim=(-1, 1),
		exclude_groups=['BLIND'],
		chance_null_metric_key='decoding_accuracy',
	)

	print('[target] sf_decoding_accuracy_by_orientation_groups.svg')
	m.plot_orientation_metric(
		metric_key='sf_decoding_by_ori',
		title='Continuous SF decoding correlation vs orientation',
		ylabel='SF decoding Pearson r',
		filename='sf_decoding_accuracy_by_orientation_groups.svg',
		exclude_groups=['BLIND'],
		chance_null_metric_key='sf_decoding_by_ori',
	)

	print('[target] orientation_decoding_accuracy_vs_cell_number_per_sf_groups.svg')
	m.plot_orientation_decoding_accuracy_vs_cell_number_per_sf()

	print('[target] orientation_decoding_accuracy_vs_cell_number_mean_non240_sf_groups.svg')
	m.plot_orientation_decoding_accuracy_vs_cell_number_mean_non240_sf()

	for n in m.DECODING_CELL_COUNTS:
		print(f'[target] orientation_decoding_accuracy_by_sf_groups_n{int(n)}.svg')
		m.plot_metric(
			metric_key=f'decoding_accuracy_n{int(n)}',
			title=f'Continuous orientation decoding correlation vs spatial frequency | N={int(n)}',
			ylabel='Circular correlation',
			filename=f'orientation_decoding_accuracy_by_sf_groups_n{int(n)}.svg',
			ylim=(-1, 1),
			exclude_groups=['BLIND'],
			chance_null_metric_key=f'decoding_accuracy_n{int(n)}',
		)

		print(f'[target] sf_decoding_accuracy_by_orientation_groups_n{int(n)}.svg')
		m.plot_orientation_metric(
			metric_key=f'sf_decoding_by_ori_n{int(n)}',
			title=f'Continuous SF decoding correlation vs orientation | N={int(n)}',
			ylabel='SF decoding Pearson r',
			filename=f'sf_decoding_accuracy_by_orientation_groups_n{int(n)}.svg',
			exclude_groups=['BLIND'],
			chance_null_metric_key=f'sf_decoding_by_ori_n{int(n)}',
		)


if __name__ == '__main__':
	main()