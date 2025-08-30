with open("GSE135222_series_matrix.txt") as f:
    for i, line in enumerate(f):
        if "!series_matrix_table_begin" in line:
            print("Table begins at line:", i)
        if "!series_matrix_table_end" in line:
            print("Table ends at line:", i)
