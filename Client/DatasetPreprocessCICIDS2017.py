import sys
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import json

load_dotenv()

def partial_shuffle_df(df, intensity, random_state=None):
    """
    Partially shuffle the DataFrame according to intensity (0 to 1):
      - 0: no shuffle (keep default order)
      - 1: full shuffle (uniform random permutation)
      - intermediate: weighted mix.
    """
    n = len(df)
    if intensity <= 0:
        return df.copy()
    if intensity >= 1:
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    indices = np.arange(n)
    # Normalize indices between 0 and 1.
    normalized = indices / (n - 1) if n > 1 else indices
    if random_state is not None:
        np.random.seed(random_state)
    random_vals = np.random.rand(n)
    # Compute the key as a weighted average of the normalized index and a random value.
    keys = (1 - intensity) * normalized + intensity * random_vals
    order = np.argsort(keys)
    return df.iloc[order].reset_index(drop=True)


def save_dataframe_in_chunks(df, filename, chunk_size=10000, desc="Saving file"):
    """
    Save a DataFrame to CSV in chunks with a progress bar.
    The header is written only once.
    """
    total = len(df)
    n_chunks = (total + chunk_size - 1) // chunk_size  # total number of chunks
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        # Write header once.
        df.iloc[:0].to_csv(f, index=False)
        for i in tqdm(range(0, total, chunk_size), desc=desc, unit="chunk", total=n_chunks):
            df.iloc[i:i + chunk_size].to_csv(f, header=False, index=False)


def plot_all_distributions(distribution_data, output_folder):
    """
    Create one combined plot that shows the training and testing label distributions
    for all clients (one row per client, two columns: Train and Test).
    For each bar, annotate the quantity and percentage above the bar.
    Save the image in the output folder.

    distribution_data: list of tuples (client_num, train_counts, test_counts)
    """
    num_clients = len(distribution_data)
    # Create subplots: 1 row per client, 2 columns.
    fig, axes = plt.subplots(num_clients, 2, figsize=(12, num_clients * 4), sharey=True)

    # Ensure axes is 2D even if there's only one client.
    if num_clients == 1:
        axes = np.array([axes])

    for i, (client_num, train_counts, test_counts) in enumerate(distribution_data):
        ax_train = axes[i, 0]
        ax_test = axes[i, 1]

        # Plot training distribution.
        bars_train = ax_train.bar(train_counts.index.astype(str), train_counts.values, color='skyblue')
        ax_train.set_title(f'Client {client_num} Train Distribution')
        ax_train.set_xlabel('Label')
        ax_train.set_ylabel('Count')
        ax_train.tick_params(axis='x', rotation=45)

        total_train = train_counts.sum()
        for bar in bars_train:
            height = bar.get_height()
            percentage = (height / total_train) * 100 if total_train > 0 else 0
            ax_train.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=8
            )

        # Plot testing distribution.
        bars_test = ax_test.bar(test_counts.index.astype(str), test_counts.values, color='salmon')
        ax_test.set_title(f'Client {client_num} Test Distribution')
        ax_test.set_xlabel('Label')
        ax_test.tick_params(axis='x', rotation=45)

        total_test = test_counts.sum()
        for bar in bars_test:
            height = bar.get_height()
            percentage = (height / total_test) * 100 if total_test > 0 else 0
            ax_test.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=8
            )

    plt.tight_layout()
    plot_filename = os.path.join(output_folder, "all_clients_distribution.png")
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename


def main():
    # Define the output folder and create it if it does not exist.
    output_folder = "CICIDS_Splitted"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Locate and read all CSV files from the "CICIDS" directory.
    data_dir = "CICIDS"  # Directory holding all CSV files.
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in directory: {data_dir}")
        return

    df_list = []
    print("Reading CSV files:")
    for file in tqdm(csv_files, desc="Reading CSV files", unit="file"):
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            tqdm.write(f"Error reading file {file}: {e}")

    # 2. Combine all files into one DataFrame.
    df_all = pd.concat(df_list, ignore_index=True)

    # 3. Identify the label column.
    # In your files, it is named "Label".
    if "Label" in df_all.columns:
        label_col = "Label"
    else:
        # Fallback: find a column with "label" in its name (case-insensitive).
        label_col_candidates = [col for col in df_all.columns if "label" in col.lower()]
        if not label_col_candidates:
            print("No label column found in the CSV files. Please check your data.")
            return
        label_col = label_col_candidates[0]
    print(f"\nUsing label column: {label_col}")

    # --- Display number of instances for each unique label type (single listing for selection) ---
    label_counts = df_all[label_col].value_counts().sort_index()
    print("\nNumber of instances for each unique label type:")
    for idx, (label, count) in enumerate(label_counts.items()):
        print(f"{idx}: {label} -> {count}")
    unique_labels = list(label_counts.index)
    # --- End of Added Section ---

    selected_idx_str = os.getenv("LABEL", "").strip()
    if selected_idx_str == "ALL":
        selected_indices = list(range(len(unique_labels)))
        selected_labels = unique_labels
    else:
        print(f"Selected index string: {selected_idx_str}")
        if not selected_idx_str:
            selected_idx_str = input("\nEnter the index(es) of the label types to keep (space separated, e.g., 0 2): ")
        try:
            selected_indices = [int(x.strip()) for x in selected_idx_str.split()]
            selected_labels = [unique_labels[i] for i in selected_indices]
        except Exception as e:
            print("Error processing the input. Make sure to enter valid indices separated by spaces.")
            return
    print(f"Selected label types: {selected_labels}")

    # 5. Filter the data to keep only rows with the selected label types.
    df_filtered = df_all[df_all[label_col].isin(selected_labels)]
    print(f"Total rows after filtering: {len(df_filtered)}")

    # 추가: 사용자가 사용할 패킷 수를 입력받고, 선택된 라벨들의 비율을 유지하며 샘플링 수행
    total_packets_str = os.getenv("TOTAL_PACKETS", "").strip()
    if not total_packets_str:
        total_packets_str = input("사용할 전체 패킷 수를 입력하세요: ")

    try:
        total_packets = int(total_packets_str)
    except ValueError:
        print("유효한 정수를 입력하세요. 전체 데이터를 사용합니다.")
        total_packets = len(df_filtered)

    # 각 라벨별 비율에 따라 샘플 개수를 결정 후 샘플링
    df_sampled_list = []
    grouped = df_filtered.groupby(label_col)
    for label, group in grouped:
        ratio = len(group) / len(df_filtered)
        n_samples = int(round(ratio * total_packets))
        # 그룹의 샘플 수가 그룹 크기를 초과하지 않도록 처리
        if n_samples > len(group):
            n_samples = len(group)
        group_sample = group.sample(n=n_samples, random_state=42)
        df_sampled_list.append(group_sample)

    df_filtered = pd.concat(df_sampled_list).reset_index(drop=True)
    print(f"샘플링 후 총 {len(df_filtered)}개의 패킷이 선택되었습니다.")

    # 선택된 라벨 정보를 JSON 파일로 저장 (파일 이름을 config.json으로 변경)
    config = {
        "selected_labels": selected_labels,
        "num_classes": len(selected_labels)  # num_classes 키에 레이블 개수 (정수) 저장
    }
    config_filename = os.path.join("..","Server", "config.json") # 파일경로
    with open(config_filename, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {config_filename}")

    # 6. Ask the user for the number of clients and client ratios.
    try:
        args = sys.argv[1:]
        if len(args) == 1:
            num_clients = int(args[0])
            print(f"Using {num_clients} clients from command line argument.")
        else:
            num_clients = int(input("\nEnter the number of clients to divide the data into: "))
    except ValueError:
        print("Invalid number of clients")
        return

    client_ratios_str = os.getenv("RATIO", "").strip()
    if not client_ratios_str:
        client_ratios_str = input("Enter the ratio for each client (space separated numbers, e.g., 1 1 1 for equal division): ")
    try:
        client_ratios = [float(x.strip()) for x in client_ratios_str.split()]
    except ValueError:
        print("Invalid input for client ratios. Please enter numbers only.")
        return

    if len(client_ratios) != num_clients:
        print("The number of ratios does not match the number of clients.")
        return

    total_ratio = sum(client_ratios)
    client_ratios = [r / total_ratio for r in client_ratios]

    # 7. Ask for training ratios for each client (all at once).
    train_ratios_str = os.getenv("TR_RATIO", "").strip()
    if not train_ratios_str:
        train_ratios_str = input("Enter the training ratio for each client (space separated numbers, e.g., 0.6 0.7 0.8): ")
    try:
        train_ratios = [float(x.strip()) for x in train_ratios_str.split()]
    except ValueError:
        print("Invalid input for training ratios. Please enter numbers only.")
        return

    if len(train_ratios) != num_clients:
        print("The number of training ratios does not match the number of clients.")
        return

    for tr in train_ratios:
        if not (0 < tr < 1):
            print("Each training ratio must be between 0 and 1 (exclusive).")
            return

    # 8. Ask for shuffle intensity.
    shuffle_intensity_str = os.getenv("SHUFFLE_INTENSITY", "").strip()
    if not shuffle_intensity_str:
        shuffle_intensity_str = input("Enter shuffle intensity (0 for no shuffle, 1 for full shuffle, e.g., 0.8): ")
    try:
        shuffle_intensity = float(shuffle_intensity_str.strip())
    except ValueError:
        print("Invalid input for shuffle intensity. Please enter a float between 0 and 1.")
        return
    if not (0 <= shuffle_intensity <= 1):
        print("Shuffle intensity must be between 0 and 1.")
        return

    # 9. Shuffle the filtered dataset using the specified intensity.
    df_filtered = partial_shuffle_df(df_filtered, shuffle_intensity, random_state=42)
    total_rows = len(df_filtered)

    # 10. Divide the dataset among the clients.
    client_dataframes = []
    start_idx = 0
    print("\nDividing the data among clients:")
    for i in range(num_clients):
        if i == num_clients - 1:
            end_idx = total_rows
        else:
            num_rows = int(client_ratios[i] * total_rows)
            end_idx = start_idx + num_rows
        client_data = df_filtered.iloc[start_idx:end_idx].copy()
        client_dataframes.append(client_data)
        print(f"Assigned {len(client_data)} rows to client {i + 1}.")
        start_idx = end_idx

    # Prepare to store splitting info for the final report.
    splitting_info_lines = []
    splitting_info_lines.append("Splitting Information")
    splitting_info_lines.append("====================")
    splitting_info_lines.append(f"Output folder: {os.path.abspath(output_folder)}")
    splitting_info_lines.append(f"Label Column Used: {label_col}")
    splitting_info_lines.append(f"Selected Labels: {selected_labels}")
    splitting_info_lines.append(f"Total rows after filtering: {len(df_filtered)}")
    splitting_info_lines.append(f"Number of Clients: {num_clients}")
    splitting_info_lines.append(f"Client Ratios (normalized): {client_ratios}")
    splitting_info_lines.append(f"Training Ratios per Client: {train_ratios}")
    splitting_info_lines.append(f"Shuffle Intensity: {shuffle_intensity}")
    splitting_info_lines.append("")

    # List to hold distribution information for all clients.
    distribution_data = []

    # 11. For each client, shuffle and then split into training and testing datasets,
    #      save each file in chunks (with a progress bar) and record distribution info.
    print("\nSplitting and saving client data:")
    for i, client_df in enumerate(client_dataframes, start=1):
        # Shuffle client's data using the specified shuffle intensity.
        client_df = partial_shuffle_df(client_df, shuffle_intensity, random_state=42)
        n = len(client_df)
        client_train_ratio = train_ratios[i - 1]
        num_train = int(client_train_ratio * n)
        train_df = client_df.iloc[:num_train]
        test_df = client_df.iloc[num_train:]

        train_filename = os.path.join(output_folder, f"client{i}_train.csv")
        test_filename = os.path.join(output_folder, f"client{i}_test.csv")
        print(f"\nClient {i}:")
        print(f"  Splitting {n} rows: {len(train_df)} for training, {len(test_df)} for testing.")

        # Save with progress bars (unit now is "chunk").
        save_dataframe_in_chunks(train_df, train_filename, desc=f"Saving client {i} train")
        save_dataframe_in_chunks(test_df, test_filename, desc=f"Saving client {i} test")

        print(f"  Saved: {train_filename} and {test_filename}")

        # Record distribution counts for this client.
        train_counts = train_df[label_col].value_counts().sort_index()
        test_counts = test_df[label_col].value_counts().sort_index()
        distribution_data.append((i, train_counts, test_counts))

        # Add client-specific info to the report.
        splitting_info_lines.append(f"Client {i}:")
        splitting_info_lines.append(f"    Total rows: {n}")
        splitting_info_lines.append(f"    Training rows: {len(train_df)}")
        splitting_info_lines.append(f"    Testing rows: {len(test_df)}")
        splitting_info_lines.append(
            f"    CSV files: {os.path.basename(train_filename)}, {os.path.basename(test_filename)}")
        splitting_info_lines.append("")

    # 12. Create one combined distribution plot for all clients.
    plot_filename = plot_all_distributions(distribution_data, output_folder)
    splitting_info_lines.append(f"Combined distribution plot: {os.path.basename(plot_filename)}")

    # 13. Write the splitting info file.
    info_filename = os.path.join(output_folder, "splitting_info.txt")
    with open(info_filename, "w", encoding="utf-8") as info_file:
        info_file.write("\n".join(splitting_info_lines))

    print("\nProcess finished successfully.")
    print(f"All outputs have been saved in: {os.path.abspath(output_folder)}")


if __name__ == "__main__":
    main()
