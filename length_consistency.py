def visualize_length_consistency(self, variability_df, output_path):
    """Create visualizations focused on complaint length consistency."""
    if len(variability_df) == 0:
        print("No data to visualize length consistency")
        return
    
    plt.figure(figsize=(16, 14))
    
    # Plot 1: Length Consistency Score vs Overall Consistency
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(
        x=variability_df['length_consistency_score'],
        y=variability_df['overall_consistency_score'],
        s=variability_df['count'] * 3,  # Size proportional to group size
        c=variability_df['lexical_diversity'],  # Color by lexical diversity
        alpha=0.7,
        cmap='viridis'
    )
    
    # Add trendline
    z = np.polyfit(variability_df['length_consistency_score'], variability_df['overall_consistency_score'], 1)
    p = np.poly1d(z)
    plt.plot(variability_df['length_consistency_score'], 
             p(variability_df['length_consistency_score']), 
             "r--", alpha=0.8)
    
    # Add correlation coefficient
    corr = variability_df['length_consistency_score'].corr(variability_df['overall_consistency_score'])
    plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), 
                 xycoords='axes fraction', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.colorbar(scatter, label='Lexical Diversity (lower is more consistent)')
    plt.title('Length Consistency vs Overall Consistency', fontsize=14, fontweight='bold')
    plt.xlabel('Length Consistency Score', fontsize=12)
    plt.ylabel('Overall Consistency Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Length Distribution Parameters for Top Groups
    plt.subplot(2, 2, 2)
    
    # Select top groups by overall consistency
    top_n = min(10, len(variability_df))
    top_groups = variability_df.sort_values('overall_consistency_score', ascending=False).head(top_n)
    
    # Create shortened labels for display
    top_groups['short_label'] = top_groups['combined_group'].apply(
        lambda x: x[:20] + "..." if len(x) > 20 else x
    )
    
    # Create bar chart of CV values
    bars = plt.barh(
        y=top_groups['short_label'],
        width=top_groups['length_cv'],
        color=plt.cm.viridis(top_groups['length_consistency_score']/100),
        alpha=0.7
    )
    
    # Add values to bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.2f}",
            va='center',
            fontsize=10
        )
    
    plt.title('Length Variability Coefficient (CV) for Top Groups', fontsize=14, fontweight='bold')
    plt.xlabel('Coefficient of Variation (lower is more consistent)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')
    plt.xlim(0, max(top_groups['length_cv']) * 1.2)
    
    # Plot 3: Boxplot of Length Distributions
    plt.subplot(2, 2, 3)
    
    # Create boxplot comparing length vs consistency
    # Group by quartiles of consistency score
    variability_df['consistency_quartile'] = pd.qcut(
        variability_df['overall_consistency_score'], 
        4, 
        labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']
    )
    
    consistency_vs_length = variability_df.groupby('consistency_quartile')['length_mean'].apply(list).reset_index()
    
    # Create boxplot data structure
    boxplot_data = []
    quartile_labels = []
    
    for _, row in consistency_vs_length.iterrows():
        boxplot_data.append(row['length_mean'])
        quartile_labels.append(row['consistency_quartile'])
    
    box = plt.boxplot(boxplot_data, patch_artist=True)
    
    # Set colors based on quartiles
    colors = ['#ff9999', '#ffcc99', '#ccff99', '#99ff99']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Complaint Length Distribution by Consistency Quartile', fontsize=14, fontweight='bold')
    plt.xticks(range(1, len(quartile_labels) + 1), quartile_labels, rotation=45)
    plt.ylabel('Average Complaint Length (characters)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot 4: Length Mean vs Length CV with Consistency as color
    plt.subplot(2, 2, 4)
    
    scatter = plt.scatter(
        x=variability_df['length_mean'],
        y=variability_df['length_cv'],
        s=variability_df['count'] * 3,
        c=variability_df['overall_consistency_score'],
        alpha=0.7,
        cmap='viridis'
    )
    
    # Add annotations for top 5 most consistent groups
    for i, row in variability_df.sort_values('overall_consistency_score', ascending=False).head(5).iterrows():
        label = row['combined_group']
        if len(label) > 15:
            label = label[:12] + '...'
        plt.annotate(
            label,
            (row['length_mean'], row['length_cv']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.colorbar(scatter, label='Overall Consistency Score')
    plt.title('Complaint Length vs Length Variability', fontsize=14, fontweight='bold')
    plt.xlabel('Average Complaint Length (characters)', fontsize=12)
    plt.ylabel('Length Coefficient of Variation', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_length_distributions(self, df, complaint_text_col, group_cols, variability_df, output_path):
    """Create detailed visualizations of complaint length distributions for selected groups."""
    if len(variability_df) <= 1:
        print("Insufficient data for length distribution comparison")
        return
    
    # Select groups for comparison
    # Get most consistent, least consistent, and a middle group
    sorted_df = variability_df.sort_values('overall_consistency_score', ascending=False)
    most_consistent = sorted_df.iloc[0]['combined_group']
    least_consistent = sorted_df.iloc[-1]['combined_group']
    
    # Try to get a middle group
    if len(sorted_df) >= 3:
        middle_idx = len(sorted_df) // 2
        middle_consistent = sorted_df.iloc[middle_idx]['combined_group']
        groups_to_compare = [most_consistent, middle_consistent, least_consistent]
        titles = ['Most Consistent', 'Medium Consistency', 'Least Consistent']
        colors = ['green', 'orange', 'red']
    else:
        groups_to_compare = [most_consistent, least_consistent]
        titles = ['Most Consistent', 'Least Consistent']
        colors = ['green', 'red']
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Length distributions comparison
    plt.subplot(2, 2, 1)
    
    # Calculate lengths for each group and plot distributions
    for i, (group, title, color) in enumerate(zip(groups_to_compare, titles, colors)):
        group_df = df[df['_combined_group'] == group]
        lengths = group_df[complaint_text_col].fillna('').apply(len)
        
        # Get kernel density estimate for smoother comparison
        sns.kdeplot(lengths, label=f"{title} (n={len(lengths)})", 
                   color=color, alpha=0.7, fill=True)
    
    plt.title('Complaint Length Distribution Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Character Count', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Word count distributions comparison
    plt.subplot(2, 2, 2)
    
    for i, (group, title, color) in enumerate(zip(groups_to_compare, titles, colors)):
        group_df = df[df['_combined_group'] == group]
        word_counts = group_df[complaint_text_col].fillna('').apply(lambda x: len(str(x).split()))
        
        sns.kdeplot(word_counts, label=f"{title} (n={len(word_counts)})", 
                   color=color, alpha=0.7, fill=True)
    
    plt.title('Word Count Distribution Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 3: Length distribution by quartile for most consistent group
    plt.subplot(2, 2, 3)
    
    most_consistent_df = df[df['_combined_group'] == most_consistent]
    most_consistent_lengths = most_consistent_df[complaint_text_col].fillna('').apply(len)
    
    # Calculate quartiles
    q1, q2, q3 = np.percentile(most_consistent_lengths, [25, 50, 75])
    
    # Plot histogram with quartile lines
    sns.histplot(most_consistent_lengths, color='green', alpha=0.6, kde=True)
    plt.axvline(q1, color='orange', linestyle='--', label=f'Q1: {int(q1)}')
    plt.axvline(q2, color='red', linestyle='-', label=f'Median: {int(q2)}')
    plt.axvline(q3, color='purple', linestyle='--', label=f'Q3: {int(q3)}')
    
    plt.title('Length Distribution - Most Consistent Group', fontsize=14, fontweight='bold')
    plt.xlabel('Character Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 4: Sentence length distribution for comparison
    plt.subplot(2, 2, 4)
    
    # Function to count sentences
    def count_sentences(text):
        if not isinstance(text, str):
            return 0
        # Simple sentence splitting on .!?
        return len(re.split(r'[.!?]+', text)) - 1  # -1 to avoid counting empty strings
    
    for i, (group, title, color) in enumerate(zip(groups_to_compare, titles, colors)):
        group_df = df[df['_combined_group'] == group]
        sentence_counts = group_df[complaint_text_col].fillna('').apply(count_sentences)
        
        sns.kdeplot(sentence_counts, label=f"{title}", 
                   color=color, alpha=0.7, fill=True)
    
    plt.title('Sentence Count Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Sentences', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
