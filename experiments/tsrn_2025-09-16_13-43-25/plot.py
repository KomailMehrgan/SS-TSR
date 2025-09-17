import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_all_weights_separate():
    """
    Finds all 'weight_*' directories, reads 'metrics.csv' from each,
    and generates four separate plots for each loss type (train/val for image/ocr).
    """
    current_directory = '.'

    # --- Create four separate figures and axes for our plots ---
    fig_train_img, ax_train_img = plt.subplots(figsize=(10, 7))
    fig_val_img, ax_val_img = plt.subplots(figsize=(10, 7))
    fig_train_ocr, ax_train_ocr = plt.subplots(figsize=(10, 7))
    fig_val_ocr, ax_val_ocr = plt.subplots(figsize=(10, 7))

    # --- Find all folders starting with 'weight_' ---
    try:
        weight_folders = [d for d in os.listdir(current_directory) if
                          os.path.isdir(os.path.join(current_directory, d)) and d.startswith('weight_')]

        if not weight_folders:
            print("Error: No folders starting with 'weight_' found.")
            print("Please place this script in the directory containing your 'weight_*' folders.")
            return

    except Exception as e:
        print(f"An error occurred while scanning directories: {e}")
        return

    # --- Loop through each folder and plot its data onto the correct axes ---
    print(f"Found and processing folders: {', '.join(sorted(weight_folders))}")
    for folder_name in sorted(weight_folders):
        try:
            # Extract the numerical value from the folder name for the legend
            weight_label = folder_name.split('_')[-1]
            csv_path = os.path.join(folder_name, 'metrics.csv')

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)

                # Plot Train Image Loss on its dedicated axes
                ax_train_img.plot(df['epoch'], df['train_img_loss'], label=f'weight={weight_label}')

                # Plot Validation Image Loss on its dedicated axes
                ax_val_img.plot(df['epoch'], df['val_img_loss'], label=f'weight={weight_label}')

                # Plot Train OCR Loss on its dedicated axes
                ax_train_ocr.plot(df['epoch'], df['train_ocr_loss'], label=f'weight={weight_label}')

                # Plot Validation OCR Loss on its dedicated axes
                ax_val_ocr.plot(df['epoch'], df['val_ocr_loss'], label=f'weight={weight_label}')
            else:
                print(f"Warning: 'metrics.csv' not found in '{folder_name}'. Skipping.")

        except Exception as e:
            print(f"An error occurred while processing '{folder_name}': {e}")

    # --- Finalize and Save the four plots ---

    # 1. Train Image Loss
    ax_train_img.set_title('Comparison of Training Image Loss', fontsize=16)
    ax_train_img.set_xlabel('Epoch')
    ax_train_img.set_ylabel('Loss')
    ax_train_img.legend()
    ax_train_img.grid(True)
    fig_train_img.tight_layout()
    fig_train_img.savefig('comparison_train_image_loss.png')
    print("✅ Saved plot to 'comparison_train_image_loss.png'")

    # 2. Validation Image Loss
    ax_val_img.set_title('Comparison of Validation Image Loss', fontsize=16)
    ax_val_img.set_xlabel('Epoch')
    ax_val_img.set_ylabel('Loss')
    ax_val_img.legend()
    ax_val_img.grid(True)
    fig_val_img.tight_layout()
    fig_val_img.savefig('comparison_val_image_loss.png')
    print("✅ Saved plot to 'comparison_val_image_loss.png'")

    # 3. Train OCR Loss
    ax_train_ocr.set_title('Comparison of Training OCR Loss', fontsize=16)
    ax_train_ocr.set_xlabel('Epoch')
    ax_train_ocr.set_ylabel('Loss')
    ax_train_ocr.legend()
    ax_train_ocr.grid(True)
    fig_train_ocr.tight_layout()
    fig_train_ocr.savefig('comparison_train_ocr_loss.png')
    print("✅ Saved plot to 'comparison_train_ocr_loss.png'")

    # 4. Validation OCR Loss
    ax_val_ocr.set_title('Comparison of Validation OCR Loss', fontsize=16)
    ax_val_ocr.set_xlabel('Epoch')
    ax_val_ocr.set_ylabel('Loss')
    ax_val_ocr.legend()
    ax_val_ocr.grid(True)
    fig_val_ocr.tight_layout()
    fig_val_ocr.savefig('comparison_val_ocr_loss.png')
    print("✅ Saved plot to 'comparison_val_ocr_loss.png'")

    # Optional: Display the plots if running on a desktop
    plt.show()


# Run the main function
if __name__ == "__main__":
    plot_all_weights_separate()