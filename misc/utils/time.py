import datetime

def print_time(str):
    print(str, '--', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def format_time_diff(start_time, stop_time):
    total_time = stop_time - start_time
    return f"{int(total_time / 3600)} hours {int((total_time % 3600) / 60)} minutes {total_time % 60:.1f} seconds"

# Print the config
def print_config(cfg):
    print()
    print(f"> Config")
    for key, value in cfg.items():
        if key in ['paths', 'tasks_size', 'task_to_subroot', 'label_to_tasks']: continue
        if key == 'tamlec_params':
            for subkey, val in value.items():
                if subkey in ['abstract_dict', 'trg_vocab', 'src_vocab', 'taxos_hector', 'taxos_tamlec']: continue
                print(f">> tamlec_{subkey}: {val}")
        else:
            print(f">> {key}: {value}")
    print()