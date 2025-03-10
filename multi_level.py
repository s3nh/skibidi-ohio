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
from datetime import datetime
import itertools
import warnings

class MultiLevelComplaintVariabilityAnalyzer:
    def __init__(self):
        """Initialize the complaint variability analyzer."""
        plt.style.use('ggplot')  # Use a business-friendly style for plots
        self.color_palette = sns.color_palette("muted")
        warnings.filterwarnings("ignore", category=UserWarning)
    
    def load_data(self, filepath):
        """Load complaint data from CSV or Excel file."""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            return pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    
    def analyze_variability(self, df, complaint_text_col, group_cols, max_groups=50, max_levels=20):
        """
        Analyze variability in complaint texts across different group combinations.
        
        Args:
            df: DataFrame containing the data
            complaint_text_col: Column name for complaint text
            group_cols: List of column names for categorical grouping
            max_groups: Maximum number of group combinations to analyze
            max_levels: Maximum number of group levels to consider
            
        Returns:
            DataFrame with variability metrics for each group combination
        """
        # Check if columns exist in the DataFrame
        for col in [complaint_text_col] + group_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the data")
        
        # Limit the number of group columns if too many
        if len(group_cols) > max_levels:
            print(f"Warning: Too many grouping columns ({len(group_cols)}). "
                  f"Limiting to the first {max_levels} columns.")
            group_cols = group_cols[:max_levels]
            
        # Generate all group combinations
        results = []
        
        # Create a concatenated group column
        df['_combined_group'] = df[group_cols].astype(str).agg(' | '.join, axis=1)
        
        # Get unique group combinations
        group_combinations = df['_combined_group'].unique()
        
        # Limit the number of groups to analyze if there are too many
        if len(group_combinations) > max_groups:
            print(f"Warning: Found {len(group_combinations)} group combinations. " 
                  f"Limiting analysis to the {max_groups} most frequent groups.")
            value_counts = df['_combined_group'].value_counts()
            group_combinations = value_counts.nlargest(max_groups).index.tolist()
        
        total_combinations = len(group_combinations)
        print(f"Analyzing {total_combinations} group combinations...")
        
        for i, combined_group in enumerate(group_combinations):
            if (i+1) % 10 == 0 or i+1 == total_combinations:
                print(f"Progress: {i+1}/{total_combinations} groups analyzed ({(i+1)/total_combinations*100:.1f}%)")
                
            group_df = df[df['_combined_group'] == combined_group]
            
            # Extract individual group values for reporting
            group_values = combined_group.split(' | ')
            group_dict = {col: val for col, val in zip(group_cols, group_values)}
            
            # Calculate basic statistics
            count = len(group_df)
            
            # Skip groups with too few samples
            if count < 2:
                continue
            
            # Calculate text similarity within group using TF-IDF and cosine similarity
            try:
                vectorizer = TfidfVectorizer().fit_transform(group_df[complaint_text_col].fillna(''))
                similarity_matrix = cosine_similarity(vectorizer)
                # Get average similarity (excluding self-similarity)
                np.fill_diagonal(similarity_matrix, 0)
                avg_similarity = similarity_matrix.sum() / (count * (count - 1)) if count > 1 else 0
            except Exception as e:
                print(f"Warning: Error calculating similarity for group {combined_group}: {str(e)}")
                avg_similarity = 0
                
            # Calculate complaint length metrics
            text_lengths = group_df[complaint_text_col].fillna('').apply(len)
            length_mean = text_lengths.mean()
            length_variance = text_lengths.var() if count > 1 else 0
            length_std = text_lengths.std() if count > 1 else 0
            length_cv = (length_std / length_mean) if length_mean > 0 else 0  # Coefficient of variation
            
            # Calculate word count statistics
            word_counts = group_df[complaint_text_col].fillna('').apply(lambda x: len(str(x).split()))
            avg_word_count = word_counts.mean()
            word_count_variance = word_counts.var() if count > 1 else 0
            
            # Calculate lexical diversity (unique words / total words)
            all_words = " ".join(group_df[complaint_text_col].fillna('').astype(str)).lower().split()
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
            
            # Create result dictionary with all group columns
            result = {
                'combined_group': combined_group,
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
            }
            
            # Add individual group values
            for col, val in group_dict.items():
                result[col] = val
                
            results.append(result)
        
        # Convert to DataFrame and sort by composite score (lower means less variability)
        if not results:
            return pd.DataFrame()
            
        results_df = pd.DataFrame(results).sort_values('composite_variability_score')
        return results_df
    
    def visualize_overview(self, variability_df, group_cols, output_path):
        """Create an executive summary visualization of variability across groups."""
        if len(variability_df) == 0:
            print("No data to visualize")
            return
            
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Overall Consistency Score (higher is better)
        plt.subplot(2, 2, 1)
        
        # Get top 15 most consistent groups (or fewer if less than 15 groups)
        top_n = min(15, len(variability_df))
        top_groups = variability_df.sort_values('overall_consistency_score', ascending=False).head(top_n)
        
        # Create simplified group labels for the chart
        top_groups['chart_label'] = top_groups['combined_group'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
        
        bars = sns.barplot(
            x='overall_consistency_score', 
            y='chart_label',
            data=top_groups,
            palette='viridis'
        )
        
        # Add value labels on bars
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.1f}',
                ha='left', 
                va='center',
                fontweight='bold'
            )
            
        plt.title(f'Top {top_n} Most Consistent Groups', fontsize=14, fontweight='bold')
        plt.xlabel('Consistency Score (higher is better)', fontsize=12)
        plt.xlim(0, 105)  # Leave room for labels
        plt.tight_layout()
        
        # Plot 2: Components Radar Chart for Top Groups
        plt.subplot(2, 2, 2)
        
        # Prepare data for radar chart
        categories = ['Similarity', 'Length Consistency', 'Content Consistency']
        
        # Select top 5 groups based on consistency score (or fewer if less than 5 groups)
        top_n = min(5, len(variability_df))
        top_radar_groups = variability_df.sort_values('overall_consistency_score', ascending=False).head(top_n)
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax = plt.subplot(2, 2, 2, polar=True)
        
        for i, (_, row) in enumerate(top_radar_groups.iterrows()):
            values = [
                row['similarity_score'],
                row['length_consistency_score'], 
                row['content_consistency_score']
            ]
            values += values[:1]  # Close the loop
            
            # Create simplified label for legend
            label = row['combined_group']
            if len(label) > 20:
                label = label[:17] + '...'
                
            ax.plot(angles, values, linewidth=2, label=label)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'])
        ax.grid(True)
        plt.title('Consistency Components (Top Groups)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', bbox_to_anchor=(1.2, -0.1))
        
        # Plot 3: Sample Size Analysis
        plt.subplot(2, 2, 3)
        
        # Get top 15 groups by sample size (or fewer if less than 15 groups)
        top_n = min(15, len(variability_df))
        top_by_size = variability_df.sort_values('count', ascending=False).head(top_n)
        top_by_size['chart_label'] = top_by_size['combined_group'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
        
        bars = sns.barplot(
            x='count', 
            y='chart_label', 
            data=top_by_size,
            palette='plasma'
        )
        
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.0f}',
                ha='left',
                va='center'
            )
            
        plt.title(f'Top {top_n} Groups by Sample Size', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Complaints', fontsize=12)
        plt.tight_layout()
        
        # Plot 4: Consistency by Hierarchical Level
        plt.subplot(2, 2, 4)
        
        # Analyze consistency by group level
        level_data = []
        
        # For each grouping column, calculate average consistency
        for i, col in enumerate(group_cols):
            # Group by this column and get mean consistency
            if col in variability_df.columns:
                group_means = variability_df.groupby(col)['overall_consistency_score'].agg(['mean', 'count'])
                for group, row in group_means.iterrows():
                    level_data.append({
                        'group_level': col,
                        'group_value': str(group),
                        'avg_consistency': row['mean'],
                        'count': row['count']
                    })
        
        if level_data:
            level_df = pd.DataFrame(level_data)
            
            # Select top values to show
            top_level_groups = []
            for level in level_df['group_level'].unique():
                level_values = level_df[level_df['group_level'] == level]
                top_values = level_values.sort_values('avg_consistency', ascending=False).head(5)
                top_level_groups.append(top_values)
                
            top_level_df = pd.concat(top_level_groups)
            
            # Create chart
            scatter = plt.scatter(
                x=top_level_df['avg_consistency'],
                y=top_level_df['group_level'] + ' | ' + top_level_df['group_value'],
                s=top_level_df['count'] * 3,
                c=top_level_df['avg_consistency'],
                alpha=0.7,
                cmap='viridis'
            )
            
            plt.colorbar(scatter, label='Consistency Score')
            plt.title('Consistency by Group Level (Top Values)', fontsize=14, fontweight='bold')
            plt.xlabel('Average Consistency Score', fontsize=12)
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, "Insufficient data for level analysis", 
                     ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_hierarchical_consistency(self, variability_df, group_cols, output_path):
        """Create a hierarchical visualization of consistency across nested groups."""
        if len(variability_df) == 0 or len(group_cols) < 2:
            print("Insufficient data for hierarchical visualization")
            return
            
        plt.figure(figsize=(16, 10))
        
        # Create a pivot table for the first two grouping levels
        first_level = group_cols[0]
        second_level = group_cols[1]
        
        if first_level not in variability_df.columns or second_level not in variability_df.columns:
            print(f"Cannot create hierarchical visualization: missing required columns")
            return
            
        try:
            pivot_data = variability_df.pivot_table(
                values='overall_consistency_score', 
                index=first_level, 
                columns=second_level, 
                aggfunc='mean'
            )
            
            # Replace NaN with 0 for visualization
            pivot_data = pivot_data.fillna(0)
            
            # Limit the size of pivot table to avoid excessive visualization
            if pivot_data.shape[0] > 20 or pivot_data.shape[1] > 20:
                print(f"Limiting hierarchical visualization to top 20 values per dimension")
                # Get top values by row and column means
                row_means = pivot_data.mean(axis=1).nlargest(20).index
                col_means = pivot_data.mean(axis=0).nlargest(20).index
                pivot_data = pivot_data.loc[row_means, col_means]
            
            # Create heatmap
            sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.1f',
                cmap='viridis',
                linewidths=0.5,
                cbar_kws={'label': 'Consistency Score'}
            )
            
            plt.title(f'Hierarchical Consistency: {first_level} × {second_level}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating hierarchical visualization: {str(e)}")
    
    def visualize_nested_treemap(self, variability_df, group_cols, output_path):
        """Create a nested treemap visualization of groups and their consistency."""
        if len(variability_df) == 0:
            print("No data for treemap visualization")
            return
            
        try:
            # Try to import the required libraries
            import plotly.express as px
            import plotly.io as pio
            
            # Prepare data for treemap
            # Use only groups with sufficient samples
            treemap_data = variability_df[variability_df['count'] >= 5].copy()
            
            if len(treemap_data) == 0:
                print("No groups with sufficient samples for treemap")
                return
                
            # Create path columns for treemap
            path_cols = group_cols.copy()
            
            # Limit path columns to first 8 to avoid excessive nesting
            if len(path_cols) > 8:
                print(f"Limiting treemap to first 8 group levels (from {len(path_cols)} total)")
                path_cols = path_cols[:8]
            
            # Create the treemap
            fig = px.treemap(
                treemap_data, 
                path=path_cols,
                values='count',
                color='overall_consistency_score',
                color_continuous_scale='viridis',
                hover_data=['count', 'avg_similarity', 'length_cv', 'lexical_diversity'],
                title='Nested Group Structure by Consistency Score'
            )
            
            fig.update_layout(
                height=800,
                width=1000,
                title_font_size=18
            )
            
            # Save as HTML and PNG
            html_path = os.path.splitext(output_path)[0] + '.html'
            pio.write_html(fig, html_path)
            fig.write_image(output_path)
            print(f"Treemap visualization saved as {output_path} and {html_path}")
        except ImportError:
            print("Could not create treemap visualization: requires plotly library")
            print("Install with: pip install plotly kaleido")
        except Exception as e:
            print(f"Error creating treemap visualization: {str(e)}")
            
    def visualize_detailed_comparison(self, variability_df, df, complaint_text_col, group_cols, output_path):
        """Create detailed comparison visualizations for the most and least consistent groups."""
        if len(variability_df) < 2:
            print("Cannot create detailed comparison: need at least 2 groups")
            return
            
        # Sort by consistency score
        sorted_df = variability_df.sort_values('overall_consistency_score', ascending=False)
        most_consistent_group = sorted_df.iloc[0]['combined_group']
        least_consistent_group = sorted_df.iloc[-1]['combined_group']
        
        # Get data for these groups
        most_df = df[df['_combined_group'] == most_consistent_group]
        least_df = df[df['_combined_group'] == least_consistent_group]
        
        plt.figure(figsize=(16, 12))
        
        # Create more descriptive labels
        most_label = f"Most Consistent: {most_consistent_group}"
        if len(most_label) > 50:  # Truncate if too long
            most_label = most_label[:47] + "..."
            
        least_label = f"Least Consistent: {least_consistent_group}"
        if len(least_label) > 50:  # Truncate if too long
            least_label = least_label[:47] + "..."
        
        # Plot 1: Length Distribution Comparison
        plt.subplot(2, 2, 1)
        most_lengths = most_df[complaint_text_col].fillna('').apply(len)
        least_lengths = least_df[complaint_text_col].fillna('').apply(len)
        
        sns.histplot(most_lengths, color='green', alpha=0.5, label=most_label)
        sns.histplot(least_lengths, color='red', alpha=0.5, label=least_label)
        plt.title('Complaint Length Distribution Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Complaint Length (characters)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Plot 2: Word Count Distribution
        plt.subplot(2, 2, 2)
        most_words = most_df[complaint_text_col].fillna('').apply(lambda x: len(str(x).split()))
        least_words = least_df[complaint_text_col].fillna('').apply(lambda x: len(str(x).split()))
        
        sns.histplot(most_words, color='green', alpha=0.5, label=most_label)
        sns.histplot(least_words, color='red', alpha=0.5, label=least_label)
        plt.title('Word Count Distribution Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Words', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Plot 3: Most Common Words - Most Consistent Group
        plt.subplot(2, 2, 3)
        all_words = " ".join(most_df[complaint_text_col].fillna('').astype(str)).lower()
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
        
        plt.title(f'Most Common Words - Most Consistent Group', fontsize=14, fontweight='bold')
        plt.xlabel('Frequency', fontsize=12)
        plt.tight_layout()
        
        # Plot 4: Most Common Words - Least Consistent Group
        plt.subplot(2, 2, 4)
        all_words = " ".join(least_df[complaint_text_col].fillna('').astype(str)).lower()
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
        
        plt.title(f'Most Common Words - Least Consistent Group', fontsize=14, fontweight='bold')
        plt.xlabel('Frequency', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def extract_template_patterns(self, df, complaint_text_col, combined_group_col, group):
        """Extract common patterns from the most consistent group."""
        group_df = df[df[combined_group_col] == group]
        
        if len(group_df) < 5:
            return {
                "avg_length": 0,
                "common_bigrams": [],
                "common_trigrams": [],
                "exemplars": list(group_df[complaint_text_col].fillna('').astype(str).values)
            }
            
        # Find average length
        avg_length = group_df[complaint_text_col].str.len().mean()
        
        # Find common words and phrases
        all_text = " ".join(group_df[complaint_text_col].fillna('').astype(str)).lower()
        all_text = re.sub(r'[^\w\s]', ' ', all_text)  # Replace punctuation with spaces
        
        # Extract frequent phrases (bigrams and trigrams)
        words = all_text.split()
        
        # Generate bigrams
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        bigram_counter = Counter(bigrams)
        common_bigrams = bigram_counter.most_common(10)
        
        # Generate trigrams
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        trigram_counter = Counter(trigrams)
        common_trigrams = trigram_counter.most_common(10)
        
        # Find exemplar complaints (closest to average length)
        group_df['length_diff'] = abs(group_df[complaint_text_col].str.len() - avg_length)
        exemplars = group_df.nsmallest(3, 'length_diff')[complaint_text_col].tolist()
        
        patterns = {
            "avg_length": int(avg_length),
            "common_bigrams": common_bigrams,
            "common_trigrams": common_trigrams,
            "exemplars": exemplars
        }
        
        return patterns

    def create_summary_report(self, variability_df, df, complaint_text_col, group_cols, output_path):
        """Create a text summary report of the variability analysis."""
        if len(variability_df) == 0:
            print("Cannot create summary report: no variability data available")
            with open(output_path, 'w') as f:
                f.write("# Multi-Level Complaint Variability Analysis Report\n\nNo data available for analysis.")
            return
            
        # Sort by consistency score
        sorted_df = variability_df.sort_values('overall_consistency_score', ascending=False)
        
        # Get most consistent group
        most_consistent_group = sorted_df.iloc[0]['combined_group']
        
        # Extract patterns from most consistent group
        patterns = self.extract_template_patterns(df, complaint_text_col, '_combined_group', most_consistent_group)
        
        # Prepare report content
        report = "# Multi-Level Complaint Variability Analysis Report\n\n"
        report += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add analysis parameters
        report += "## Analysis Parameters\n\n"
        report += f"- Text column: `{complaint_text_col}`\n"
        report += f"- Grouping columns: `{', '.join(group_cols)}`\n"
        report += f"- Total group combinations analyzed: {len(variability_df)}\n\n"
        
        # Top 10 most consistent groups
        report += "## Top 10 Most Consistent Groups\n\n"
        report += "| Group | Consistency Score | Sample Size | Similarity | Length CV | Lexical Diversity |\n"
        report += "|-------|------------------|-------------|------------|-----------|-------------------|\n"
        
        for _, row in sorted_df.head(10).iterrows():
            report += f"| {row['combined_group']} | {row['overall_consistency_score']:.1f} | {row['count']} "
            report += f"| {row['avg_similarity']:.2f} | {row['length_cv']:.2f} | {row['lexical_diversity']:.2f} |\n"
        
        # Consistency by hierarchical level
        report += "\n\n## Consistency Analysis by Grouping Level\n\n"
        
        for level in group_cols:
            if level in variability_df.columns:
                report += f"### Level: {level}\n\n"
                report += "| Value | Avg. Consistency | Sample Size |\n"
                report += "|-------|-----------------|-------------|\n"
                
                # Group by this level and calculate average consistency
                level_stats = variability_df.groupby(level)['overall_consistency_score'].agg(['mean', 'count'])
                level_stats = level_stats.sort_values('mean', ascending=False)
                
                # Limit to top 20 values if there are too many
                if len(level_stats) > 20:
                    level_stats = level_stats.head(20)
                    report += "*Note: Showing top 20 values by consistency score*\n\n"
                
                for value, stats in level_stats.iterrows():
                    report += f"| {value} | {stats['mean']:.1f} | {stats['count']} |\n"
                
                report += "\n"
        
        report += "\n\n## Key Insights\n\n"
        
        # Add insights about most consistent group
        most_consistent = sorted_df.
