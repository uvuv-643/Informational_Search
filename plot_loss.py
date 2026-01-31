import matplotlib.pyplot as plt

def parse_data(filepath):
    batches = []
    losses = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('batch'):
                continue
            
            try:
                parts = line.split()
                # Format: batch <number> loss: <number>
                # parts: ['batch', '100', 'loss:', '1.618...']
                batch_num = int(parts[1])
                loss_val = float(parts[3])
                
                batches.append(batch_num)
                losses.append(loss_val)
            except (IndexError, ValueError) as e:
                print(f"Skipping malformed line: {line}")
                continue
                
    return batches, losses

def plot_loss(batches, losses, output_file='loss_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(batches, losses, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    data_file = "data.txt"
    batches, losses = parse_data(data_file)
    
    if batches:
        plot_loss(batches, losses)
    else:
        print("No data found to plot.")
