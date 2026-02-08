import pandas as pd

class MetricsHandler:
    '''
    Buffered CSV writer for experiment metrics.

    Rows are accumulated in memory and flushed to disk in batches to reduce
    write overhead.
    '''
    def __init__(self, columns, output_path, flush_every=10):
        '''
        Initialize a metrics buffer and create a CSV with headers.

        :param columns: Iterable of column names for the CSV.
        :param output_path: Path to the CSV file to create/append.
        :param flush_every: Flush to disk after this many buffered rows.
        :return: None.
        '''
        self._pd = pd  # keep local reference to avoid module unloading issue
        self.buffer = []
        self.columns = columns
        self.output_path = output_path
        self.flush_every = flush_every
        # initialize empty CSV with headers
        self._pd.DataFrame(columns=columns).to_csv(output_path, index=False)

    def add_row(self, new_row_dict):
        '''
        Add a metrics row to the buffer and flush if needed.

        :param new_row_dict: Mapping of column names to values.
        :return: None.
        '''
        assert isinstance(new_row_dict, dict)
        self.buffer.append(new_row_dict)
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        '''
        Flush buffered rows to the CSV file.

        :return: None.
        '''
        if not self.buffer:
            return
        df = self._pd.DataFrame(self.buffer)
        df.to_csv(self.output_path, mode='a', header=False, index=False)
        self.buffer.clear()

    def __del__(self):
        """Ensure remaining buffered rows are flushed before destruction."""
        try:
            self.flush()
        except Exception:
            # Avoid any crash during interpreter shutdown
            pass
