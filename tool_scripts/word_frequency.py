import re
from collections import Counter
from pathlib import Path

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.plot._backend import matplotlib, pyplot as plt

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

PROJECT_ROOT = infer_project_root()
DATA_PATH = PROJECT_ROOT / 'data' / 'time_machine' / 'timemachine.txt'
OUTPUT_PATH = PROJECT_ROOT / 'output' / 'word_frequency.png'

def load_and_count(path: Path, top_n: int = 10):
    text = path.read_text(encoding='utf-8').lower()
    words = re.findall(r'[a-z]+', text)
    return Counter(words).most_common(top_n)

def plot_bar(word_counts, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    words, counts = zip(*word_counts)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(words, counts, color=plt.cm.viridis(
        [i / len(words) for i in range(len(words))]))

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(count),
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title('Top 10 Most Frequent Words in The Time Machine',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Word', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_ylim(0, max(counts) * 1.12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=11)

    plt.tight_layout()
    fig.savefig(output, dpi=150)
    print(f'Chart saved to: {output}')
    try:
        plt.show()
    except Exception:
        pass

def main():
    print(f'Reading file: {DATA_PATH}')
    word_counts = load_and_count(DATA_PATH)

    print('\nTop 10 Most Frequent Words:')
    print(f'{"Rank":<6}{"Word":<20}{"Count":<10}')
    print('-' * 36)
    for rank, (word, count) in enumerate(word_counts, 1):
        print(f'{rank:<6}{word:<20}{count:<10}')

    plot_bar(word_counts, OUTPUT_PATH)

if __name__ == '__main__':
    main()
