import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import spacy
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns

class ComplaintTemplateExtractor:
    def __init__(self):
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Installing spaCy model...")
            import subprocess
            subprocess.call("python -m spacy download en_core_web_sm", shell=True)
            self.nlp = spacy.load("en_core_web_sm")
    
    def load_data(self, filepath):
        """Load complaint data from CSV or Excel file."""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            return pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    
    def analyze_variance(self, df, text_col, group_col, answer_col):
        """
        Analyze variance in the answer column across different groups.
        
        Args:
            df: DataFrame containing the data
            text_col: Column name for complaint text
            group_col: Column name for categorical grouping
            answer_col: Column name for the final answer
            
        Returns:
            DataFrame with variance metrics for each group
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
                vectorizer = TfidfVectorizer().fit_transform(group_df[answer_col])
                similarity_matrix = cosine_similarity(vectorizer)
                # Get average similarity (excluding self-similarity)
                np.fill_diagonal(similarity_matrix, 0)
                avg_similarity = similarity_matrix.sum() / (count * (count - 1))
            else:
                avg_similarity = 0
                
            # Calculate answer length variance
            answer_lengths = group_df[answer_col].apply(len)
            length_variance = answer_lengths.var() if count > 1 else 0
            
            # Calculate lexical diversity (unique words / total words)
            total_words = sum([len(ans.split()) for ans in group_df[answer_col]])
            unique_words = len(set(" ".join(group_df[answer_col].tolist()).lower().split()))
            lexical_diversity = unique_words / total_words if total_words > 0 else 0
            
            # Composite variance score (lower is more consistent)
            # Higher similarity and lower lexical diversity indicate more consistency
            composite_score = (1 - avg_similarity) + (0.5 * lexical_diversity) + (0.0001 * length_variance)
            
            results.append({
                'group': group,
                'count': count,
                'avg_similarity': avg_similarity,
                'length_variance': length_variance,
                'lexical_diversity': lexical_diversity,
                'composite_score': composite_score  # Lower score means more consistency
            })
        
        results_df = pd.DataFrame(results).sort_values('composite_score')
        return results_df
    
    def extract_template(self, df, text_col, group_col, answer_col, selected_group, n_templates=1):
        """
        Extract template patterns from the most consistent group.
        
        Args:
            df: DataFrame containing the data
            text_col: Column name for complaint text
            group_col: Column name for categorical grouping
            answer_col: Column name for the final answer
            selected_group: The group to extract templates from
            n_templates: Number of templates to extract
            
        Returns:
            List of extracted template strings
        """
        group_df = df[df[group_col] == selected_group]
        answers = group_df[answer_col].tolist()
        
        # Process all answers with spaCy
        processed_answers = [self.nlp(answer) for answer in answers]
        
        # Extract common phrases (n-grams)
        templates = []
        
        # Method 1: Find common sentence structures
        sentence_patterns = []
        for doc in processed_answers:
            # Create a pattern based on POS tags and entities
            pattern = []
            for sent in doc.sents:
                sent_pattern = []
                for token in sent:
                    if token.ent_type_:  # If it's a named entity
                        sent_pattern.append(f"[{token.ent_type_}]")
                    elif not token.is_punct and not token.is_stop:
                        sent_pattern.append(token.lemma_)
                    else:
                        sent_pattern.append(token.text)
                sentence_patterns.append(" ".join(sent_pattern))
        
        # Count sentence patterns and get most common
        pattern_counter = Counter(sentence_patterns)
        common_patterns = pattern_counter.most_common(n_templates)
        
        # Method 2: Word frequency analysis
        # Find words that appear in over 70% of answers
        word_counts = {}
        for doc in processed_answers:
            for token in doc:
                if not token.is_stop and not token.is_punct and len(token.text) > 2:
                    word_counts[token.lemma_] = word_counts.get(token.lemma_, 0) + 1
        
        common_words = {word: count for word, count in word_counts.items() 
                        if count >= 0.7 * len(answers)}
        
        # Create a template using most common sentences and enriched with common words
        for pattern, count in common_patterns:
            templates.append({
                'template': pattern,
                'frequency': count,
                'common_words': list(common_words.keys())[:10]  # Top 10 common words
            })
            
        # Method 3: Find the most representative answer
        if not templates and len(answers) > 0:
            # If no consistent patterns found, use the answer closest to average length
            avg_length = sum(len(a) for a in answers) / len(answers)
            best_answer = min(answers, key=lambda x: abs(len(x) - avg_length))
            templates.append({
                'template': best_answer,
                'frequency': 1,
                'common_words': list(common_words.keys())[:10]
            })
        
        return templates
    
    def create_gemini_prompt(self, templates, complaint_text):
        """
        Create a Gemini prompt from the extracted templates.
        
        Args:
            templates: List of template dictionaries
            complaint_text: Example complaint text to include in prompt
            
        Returns:
            String containing the Gemini prompt
        """
        if not templates:
            return "Unable to generate a prompt template due to insufficient consistent patterns."
        
        template = templates[0]  # Use the most common template
        
        prompt = f"""
You are analyzing customer complaints to provide standardized responses.

COMPLAINT:
{complaint_text}

Your task is to generate a response following this pattern:
{template['template']}

Key concepts to include in your response:
{', '.join(template['common_words'])}

Please provide a professional, empathetic response that addresses the specific issues raised in the complaint.
"""
        return prompt
    
    def visualize_results(self, variance_df, output_path=None):
        """
        Visualize the variance analysis results.
        
        Args:
            variance_df: DataFrame with variance metrics
            output_path: Path to save visualization
        """
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Composite Score by Group (lower is better)
        plt.subplot(2, 2, 1)
        sns.barplot(x='group', y='composite_score', data=variance_df)
        plt.title('Consistency Score by Group (Lower is Better)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Plot 2: Answer Similarity
        plt.subplot(2, 2, 2)
        sns.barplot(x='group', y='avg_similarity', data=variance_df)
        plt.title('Answer Similarity by Group (Higher is Better)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Plot 3: Sample Count
        plt.subplot(2, 2, 3)
        sns.barplot(x='group', y='count', data=variance_df)
        plt.title('Sample Count by Group')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Plot 4: Lexical Diversity
        plt.subplot(2, 2, 4)
        sns.barplot(x='group', y='lexical_diversity', data=variance_df)
        plt.title('Lexical Diversity (Lower is More Consistent)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to {output_path}")
        else:
            plt.show()
    
    def run(self, filepath, text_col, group_col, answer_col, output_dir='./', example_complaint=None):
        """
        Run the complete template extraction process.
        
        Args:
            filepath: Path to the data file
            text_col: Column name for complaint text
            group_col: Column name for categorical grouping
            answer_col: Column name for the final answer
            output_dir: Directory to save outputs
            example_complaint: Example complaint text for prompt generation
            
        Returns:
            Dictionary with analysis results and extracted templates
        """
        print("Loading data...")
        df = self.load_data(filepath)
        
        print("Analyzing variance across groups...")
        variance_df = self.analyze_variance(df, text_col, group_col, answer_col)
        
        # Get the group with lowest variance score
        best_group = variance_df.iloc[0]['group']
        print(f"Group with highest consistency: {best_group}")
        
        print("Extracting templates from consistent group...")
        templates = self.extract_template(df, text_col, group_col, answer_col, best_group)
        
        # Create a Gemini prompt
        if example_complaint is None and len(df) > 0:
            # Use a random complaint from the dataset as an example
            example_complaint = df[text_col].sample(1).iloc[0]
        
        gemini_prompt = self.create_gemini_prompt(templates, example_complaint)
        
        # Save results
        results = {
            'variance_analysis': variance_df.to_dict('records'),
            'best_group': best_group,
            'extracted_templates': templates,
            'gemini_prompt': gemini_prompt
        }
        
        # Save to JSON
        with open(f"{output_dir}/template_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations
        self.visualize_results(variance_df, f"{output_dir}/variance_analysis.png")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Extract templates from complaint data")
    parser.add_argument("--file", required=True, help="Path to CSV or Excel file with complaint data")
    parser.add_argument("--text_col", required=True, help="Column name for complaint text")
    parser.add_argument("--group_col", required=True, help="Column name for categorical grouping")
    parser.add_argument("--answer_col", required=True, help="Column name for final answers")
    parser.add_argument("--output_dir", default="./", help="Directory to save outputs")
    parser.add_argument("--example", default=None, help="Example complaint text for prompt generation")
    
    args = parser.parse_args()
    
    extractor = ComplaintTemplateExtractor()
    results = extractor.run(
        args.file, 
        args.text_col, 
        args.group_col, 
        args.answer_col, 
        args.output_dir,
        args.example
    )
    
    print("\nExtracted Gemini Prompt Template:")
    print("="*80)
    print(results['gemini_prompt'])
    print("="*80)
    
    print(f"\nFull analysis results saved to {args.output_dir}/template_results.json")

if __name__ == "__main__":
    main()
