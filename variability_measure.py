import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import argparse
import json
import os

class ComplaintVariabilityAnalyzer:
    def __init__(self):
        """Initialize the complaint variability analyzer."""
        plt.style.use('ggplot')  # Use a business-friendly style for plots
        self.color_palette = sns.color_palette("muted")
    
    def load_data(self, filepath):
        """Load complaint data from CSV or Excel file."""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            return pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    
    def analyze_variability(self, df, text_col, group_col, answer_col):
        """
        Analyze variability in the answer column across different groups.
        
        Args:
            df: DataFrame containing the data
            text_col: Column name for complaint text
            group_col: Column name for categorical grouping
            answer_col: Column name for the final answer
            
        Returns:
            DataFrame with variability metrics for each group
        """
        # Get unique groups
        groups = df[group_col].unique()
        
        results = []
        
        for group in groups:
            group_df = df[df[group_col] == group]
            
            # Calculate basic statistics
            count = len(group_df)
            
            # Calculate text similarity within group using TF-IDF and cosine similarity
            if count > 1:
                try:
                    vectorizer = TfidfVectorizer().fit_transform(group_df[answer_col].fillna(''))
                    similarity_matrix = cosine_similarity(vectorizer)
                    # Get average similarity (excluding self-similarity)
                    np.fill_diagonal(similarity_matrix, 0)
                    avg_similarity = similarity_matrix.sum() / (count * (count - 1)) if count > 1 else 0
                except:
                    # Handle cases where text vectorization fails
                    avg_similarity = 0
            else:
                avg_similarity = 0
                
            # Calculate answer length metrics
            answer_lengths = group_df[answer_col].fillna('').apply(len)
            length_mean = answer_lengths.mean()
            length_variance = answer_lengths.var() if count > 1 else 0
            length_std = answer_lengths.std() if count > 1 else 0
            length_cv = (length_std / length_mean) if length_mean > 0 else 0  # Coefficient of variation
            
            # Calculate word count statistics
            word_counts = group_df[answer_col].fillna('').apply(lambda x: len(str(x).split()))
            avg_word_count = word_counts.mean()
            word_count_variance = word_counts.var() if count > 1 else 0
            
            # Calculate lexical diversity (unique words / total words)
            all_words = " ".join(group_df[answer_col].fillna('').astype(str)).lower().split()
            total_words = len(all_words) if all_words else 1  # Avoid division by zero
            unique_words = len(set(all_words))
            lexical_diversity = unique_words / total_words
            
            # Word patterns - identify common phrases/words
            word_counter = Counter(all_words)
            most_common_words = word_counter.most_common(5) if word_counter else []
            
            # Composite variability score (lower is more consistent)
            # Components weighted to create a 0-100 scale with lower values meaning less variability
            similarity_component = (1 - avg_similarity) * 40  # 0-40 points
            length_component = min(length_cv * 30, 30)  # 0-30 points
            diversity_component = lexical_diversity * 30  # 0-30 points
            
            composite_score = similarity_component + length_component + diversity_component
            
            results.append({
                'group': group,
                'count': count,
                'avg_similarity': avg_similarity,
                'similarity_score': 100 - (similarity_component * 2.5),  # Convert to 0-100 scale
                'length_mean': length_mean,
                'length_variance': length_variance,
                'length_cv': length_cv,
                'length_consistency_score': 100 - (length_component * 3.33),  # Convert to 0-100 scale
                'lexical_diversity': lexical_diversity,
                'content_consistency_score': 100 - (diversity_component * 3.33),  # Convert to 0-100 scale
                'avg_word_count': avg_word_count,
                'word_count_variance': word_count_variance,
                'most_common_words': most_common_words,
                'composite_variability_score': composite_score,
                'overall_consistency_score': 100 - composite_score  # Higher is better
            })
        
        # Convert to DataFrame and sort by composite score (lower means less variability)
        results_df = pd.DataFrame(results).sort_values('composite_variability_score')
        return results_df
    
    def visualize_overview(self, variability_df, output_path):
        """Create an executive summary visualization of variability across groups."""
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Overall Consistency Score (higher is better)
        plt.subplot(2, 2, 1)
        bars = sns.barplot(
            x='group', 
            y='overall_consistency_score', 
            data=variability_df.sort_values('overall_consistency_score', ascending=False),
            palette='viridis'
        )
        
        # Add value labels on bars
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 1,
                f'{bar.get_height():.1f}',
                ha='center', 
                va='bottom',
                fontweight='bold'
            )
            
        plt.title('Overall Consistency Score by Group', fontsize=14, fontweight='bold')
        plt.ylabel('Consistency Score (higher is better)', fontsize=12)
        plt.xlabel('Group', fontsize=12)
        plt.ylim(0, 105)  # Leave room for labels
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Components Radar Chart
        plt.subplot(2, 2, 2)
        
        # Prepare data for radar chart
        categories = ['Similarity', 'Length Consistency', 'Content Consistency']
        
        # Select top 5 groups based on consistency score
        top_groups = variability_df.sort_values('overall_consistency_score', ascending=False).head(5)
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax = plt.subplot(2, 2, 2, polar=True)
        
        for i, (_, row) in enumerate(top_groups.iterrows()):
            values = [
                row['similarity_score'],
                row['length_consistency_score'], 
                row['content_consistency_score']
            ]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=row['group'])
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'])
        ax.grid(True)
        plt.title('Consistency Components (Top 5 Groups)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', bbox_to_anchor=(1.2, -0.1))
        
        # Plot 3: Group Sizes
        plt.subplot(2, 2, 3)
        bars = sns.barplot(
            x='group', 
            y='count', 
            data=variability_df.sort_values('count', ascending=False),
            palette='plasma'
        )
        
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.3,
                f'{bar.get_height():.0f}',
                ha='center', 
                va='bottom'
            )
            
        plt.title('Sample Size by Group', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Complaints', fontsize=12)
        plt.xlabel('Group', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 4: Length Variability
        plt.subplot(2, 2, 4)
        scatter = plt.scatter(
            x=variability_df['length_mean'], 
            y=variability_df['length_cv'],
            s=variability_df['count'] * 10,  # Size represents sample count
            c=variability_df['overall_consistency_score'],  # Color represents consistency
            alpha=0.7,
            cmap='viridis',
        )
        
        # Add group labels to points
        for i, row in variability_df.iterrows():
            plt.annotate(
                row['group'], 
                (row['length_mean'], row['length_cv']), 
                xytext=(7, 0), 
                textcoords='offset points'
            )
        
        plt.colorbar(scatter, label='Consistency Score')
        plt.title('Length Variability by Group', fontsize=14, fontweight='bold')
        plt.xlabel('Average Length (chars)', fontsize=12)
        plt.ylabel('Coefficient of Variation (lower is more consistent)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_detailed_comparison(self, variability_df, df, group_col, answer_col, output_path):
        """Create detailed comparison visualizations for the most and least consistent groups."""
        # Sort by consistency score
        sorted_df = variability_df.sort_values('overall_consistency_score', ascending=False)
        most_consistent_group = sorted_df.iloc[0]['group']
        least_consistent_group = sorted_df.iloc[-1]['group']
        
        # Get data for these groups
        most_df = df[df[group_col] == most_consistent_group]
        least_df = df[df[group_col] == least_consistent_group]
        
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Length Distribution Comparison
        plt.subplot(2, 2, 1)
        most_lengths = most_df[answer_col].fillna('').apply(len)
        least_lengths = least_df[answer_col].fillna('').apply(len)
        
        sns.histplot(most_lengths, color='green', alpha=0.5, label=f'{most_consistent_group} (Most Consistent)')
        sns.histplot(least_lengths, color='red', alpha=0.5, label=f'{least_consistent_group} (Least Consistent)')
        plt.title('Answer Length Distribution Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Answer Length (characters)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        
        # Plot 2: Word Count Distribution
        plt.subplot(2, 2, 2)
        most_words = most_df[answer_col].fillna('').apply(lambda x: len(str(x).split()))
        least_words = least_df[answer_col].fillna('').apply(lambda x: len(str(x).split()))
        
        sns.histplot(most_words, color='green', alpha=0.5, label=f'{most_consistent_group} (Most Consistent)')
        sns.histplot(least_words, color='red', alpha=0.5, label=f'{least_consistent_group} (Least Consistent)')
        plt.title('Word Count Distribution Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Words', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        
        # Plot 3: Most Common Words - Most Consistent Group
        plt.subplot(2, 2, 3)
        all_words = " ".join(most_df[answer_col].fillna('').astype(str)).lower()
        all_words = re.sub(r'[^\w\s]', '', all_words)  # Remove punctuation
        word_counter = Counter(all_words.split())
        common_words = pd.DataFrame(word_counter.most_common(15), columns=['Word', 'Frequency'])
        
        bars = sns.barplot(
            x='Frequency', 
            y='Word', 
            data=common_words,
            palette='Greens_d'
        )
        
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.0f}',
                va='center'
            )
        
        plt.title(f'Most Common Words - {most_consistent_group} Group', fontsize=14, fontweight='bold')
        plt.xlabel('Frequency', fontsize=12)
        plt.tight_layout()
        
        # Plot 4: Most Common Words - Least Consistent Group
        plt.subplot(2, 2, 4)
        all_words = " ".join(least_df[answer_col].fillna('').astype(str)).lower()
        all_words = re.sub(r'[^\w\s]', '', all_words)  # Remove punctuation
        word_counter = Counter(all_words.split())
        common_words = pd.DataFrame(word_counter.most_common(15), columns=['Word', 'Frequency'])
        
        bars = sns.barplot(
            x='Frequency', 
            y='Word', 
            data=common_words,
            palette='Reds_d'
        )
        
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.0f}',
                va='center'
            )
        
        plt.title(f'Most Common Words - {least_consistent_group} Group', fontsize=14, fontweight='bold')
        plt.xlabel('Frequency', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_similarity_heatmap(self, df, group_col, answer_col, group, output_path):
        """Create a similarity heatmap for a specific group."""
        group_df = df[df[group_col] == group]
        
        if len(group_df) > 50:
            # For large groups, sample to avoid excessive computation
            group_df = group_df.sample(50, random_state=42)
        
        # Calculate similarity matrix
        try:
            vectorizer = TfidfVectorizer().fit_transform(group_df[answer_col].fillna(''))
            similarity_matrix = cosine_similarity(vectorizer)
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                similarity_matrix, 
                cmap='viridis', 
                vmin=0, 
                vmax=1,
                xticklabels=False,
                yticklabels=False
            )
            plt.title(f'Answer Similarity Heatmap - {group} Group', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            return True
        except:
            print(f"Could not generate similarity heatmap for {group} group")
            return False

    def create_summary_report(self, variability_df, output_path):
        """Create a text summary report of the variability analysis."""
        # Sort by consistency score
        sorted_df = variability_df.sort_values('overall_consistency_score', ascending=False)
        
        # Prepare report content
        report = "# Complaint Variability Analysis Report\n\n"
        report += "## Summary of Groups by Consistency\n\n"
        report += "| Group | Consistency Score | Sample Size | Similarity | Length CV | Lexical Diversity |\n"
        report += "|-------|------------------|-------------|------------|-----------|-------------------|\n"
        
        for _, row in sorted_df.iterrows():
            report += f"| {row['group']} | {row['overall_consistency_score']:.1f} | {row['count']} "
            report += f"| {row['avg_similarity']:.2f} | {row['length_cv']:.2f} | {row['lexical_diversity']:.2f} |\n"
        
        report += "\n\n## Key Insights\n\n"
        
        # Add insights
        most_consistent = sorted_df.iloc[0]
        least_consistent = sorted_df.iloc[-1]
        
        report += f"### Most Consistent Group: {most_consistent['group']}\n"
        report += f"- Overall consistency score: {most_consistent['overall_consistency_score']:.1f}/100\n"
        report += f"- Sample size: {most_consistent['count']} complaints\n"
        report += f"- Text similarity: {most_consistent['avg_similarity']:.2f} (higher values indicate more similarity)\n"
        report += f"- Length consistency: CV = {most_consistent['length_cv']:.2f} (lower values indicate more consistency)\n"
        report += f"- Lexical diversity: {most_consistent['lexical_diversity']:.2f} (lower values indicate more consistency)\n"
        report += "- Common words: "
        for word, count in most_consistent['most_common_words']:
            report += f"{word} ({count}), "
        report = report.rstrip(", ") + "\n\n"
        
        report += f"### Least Consistent Group: {least_consistent['group']}\n"
        report += f"- Overall consistency score: {least_consistent['overall_consistency_score']:.1f}/100\n"
        report += f"- Sample size: {least_consistent['count']} complaints\n"
        report += f"- Text similarity: {least_consistent['avg_similarity']:.2f} (higher values indicate more similarity)\n"
        report += f"- Length consistency: CV = {least_consistent['length_cv']:.2f} (lower values indicate more consistency)\n"
        report += f"- Lexical diversity: {least_consistent['lexical_diversity']:.2f} (lower values indicate more consistency)\n"
        report += "- Common words: "
        for word, count in least_consistent['most_common_words']:
            report += f"{word} ({count}), "
        report = report.rstrip(", ") + "\n\n"
        
        # Add business recommendations
        report += "## Business Recommendations\n\n"
        report += f"1. **Template Development**: Consider using the '{most_consistent['group']}' group responses as a basis for "
        report += "developing standardized templates, as this group shows the most consistent response patterns.\n\n"
        
        report += f"2. **Training Opportunity**: The '{least_consistent['group']}' group shows significant variability in responses. "
        report += "This suggests an opportunity for standardization training or clearer response guidelines for this category.\n\n"
        
        report += "3. **Response Benchmarking**: Use the consistency scores as benchmarks to track improvement in response standardization over time.\n\n"
        
        # Write report to file
        with open(output_path, 'w') as f:
            f.write(report)
    
    def run(self, filepath, text_col, group_col, answer_col, output_dir='./'):
        """
        Run the complete variability analysis process.
        
        Args:
            filepath: Path to the data file
            text_col: Column name for complaint text
            group_col: Column name for categorical grouping
            answer_col: Column name for the final answer
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with analysis results
        """
        print("Loading data...")
        df = self.load_data(filepath)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print("Analyzing variability across groups...")
        variability_df = self.analyze_variability(df, text_col, group_col, answer_col)
        
        # Get the group with lowest variability score
        best_group = variability_df.iloc[0]['group']
        print(f"Group with highest consistency: {best_group}")
        
        # Generate visualizations
        print("Generating visualizations...")
        
        # Overview visualization
        self.visualize_overview(
            variability_df, 
            os.path.join(output_dir, "variability_overview.png")
        )
        
        # Detailed comparison
        self.visualize_detailed_comparison(
            variability_df, 
            df, 
            group_col, 
            answer_col,
            os.path.join(output_dir, "detailed_comparison.png")
        )
        
        # Similarity heatmaps for most and least consistent groups
        most_consistent_group = variability_df.sort_values('overall_consistency_score', ascending=False).iloc[0]['group']
        least_consistent_group = variability_df.sort_values('overall_consistency_score', ascending=False).iloc[-1]['group']
        
        self.visualize_similarity_heatmap(
            df,
            group_col,
            answer_col,
            most_consistent_group,
            os.path.join(output_dir, f"similarity_heatmap_most_consistent.png")
        )
        
        self.visualize_similarity_heatmap(
            df,
            group_col,
            answer_col,
            least_consistent_group,
            os.path.join(output_dir, f"similarity_heatmap_least_consistent.png")
        )
        
        # Create summary report
        self.create_summary_report(
            variability_df,
            os.path.join(output_dir, "variability_report.md")
        )
        
        # Save detailed results to CSV
        variability_df.drop('most_common_words', axis=1).to_csv(
            os.path.join(output_dir, "variability_metrics.csv"), 
            index=False
        )
        
        # Save results to JSON (excluding non-serializable objects)
        results = {
            'variability_analysis': variability_df.drop('most_common_words', axis=1).to_dict('records'),
            'most_consistent_group': most_consistent_group,
            'least_consistent_group': least_consistent_group,
        }
        
        with open(os.path.join(output_dir, "variability_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis complete! Results saved to {output_dir}")
        return results

def main():
    parser = argparse.ArgumentParser(description="Analyze variability in complaint responses")
    parser.add_argument("--file", required=True, help="Path to CSV or Excel file with complaint data")
    parser.add_argument("--text_col", required=True, help="Column name for complaint text")
    parser.add_argument("--group_col", required=True, help="Column name for categorical grouping")
    parser.add_argument("--answer_col", required=True, help="Column name for final answers")
    parser.add_argument("--output_dir", default="./results", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    analyzer = ComplaintVariabilityAnalyzer()
    results = analyzer.run(
        args.file, 
        args.text_col, 
        args.group_col, 
        args.answer_col, 
        args.output_dir
    )
    
    print(f"\nAnalysis complete! Most consistent group: {results['most_consistent_group']}")
    print(f"Full analysis results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
