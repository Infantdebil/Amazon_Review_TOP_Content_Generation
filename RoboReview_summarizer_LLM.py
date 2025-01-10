#!/usr/bin/env python
# coding: utf-8

# # Part III: Product Review Summary
# 
# Generative AI Summaries: Use Large Language Models (LLMs) to summarize reviews and generate articles recommending the top products in each category. Classical summarization techniques are intentionally avoided for this task to leverage the power of state-of-the-art LLMs.
# 
# 
# Create a model which, for each product category, generates a short article, like a blogpost reviewer would write, to help customers choose the best one for them.
# Here’s a suggestion of what the model may output
# The Top 3 products and key differences between them. When should a consumer choose one or another?
# Top complaints for each of those products
# What is the worst product in the category and why you should never buy it
# More ideas: Look at the Consumer Reviews website, The Verge, The Wirecutter…

# 

# ### 1. Setup & Initialisation

# In[1]:


get_ipython().system('pip install cohere')

import torch
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
import os
from dotenv import load_dotenv
import cohere
from sklearn.feature_extraction.text import CountVectorizer


# ### 2. Loading the pre-processed Data 
# 
# Processed and Sentiment added and clustered

# In[2]:


file_path = 'dataset_with_ sentiment_clustered.csv'  # loading file from directory 
data = pd.read_csv(file_path)


# In[3]:


data.head(2)


# In[4]:


data.columns


# In[5]:


# Fix: Convert the 'reviews.date' column to datetime
data['reviews.date'] = pd.to_datetime(data['reviews.date'], errors='coerce')  # Convert and handle invalid dates

# Step 1: Define helper functions
def combine_reviews_by_sentiment(df, sentiment_range, max_reviews=10):
    """
    Combine up to `max_reviews` (title + text) for a specific sentiment range.
    """
    filtered_reviews = df[(df['sentiment'] >= sentiment_range[0]) & (df['sentiment'] <= sentiment_range[1])]
    filtered_reviews = filtered_reviews.head(max_reviews)  # Limit to top `max_reviews`
    combined_reviews = filtered_reviews['reviews.title'].fillna('') + " " + filtered_reviews['reviews.text'].fillna('')
    return " ".join(combined_reviews)

def extract_top_keywords(text, n=5):
    """
    Extract the top `n` keywords from the text using CountVectorizer.
    """
    if not text.strip():  # Check if the text is empty or contains only whitespace
        return ""
    vectorizer = CountVectorizer(max_features=n, stop_words='english')
    X = vectorizer.fit_transform([text])
    return ", ".join(vectorizer.get_feature_names_out())

# Step 2: Group data by product and calculate metrics
grouped_data = data.groupby(['name', 'category_cluster', 'cluster_name']).agg(
    mean_sentiment=('sentiment', 'mean'),  # Calculate the mean sentiment
    review_count=('reviews.text', 'count'),  # Count the number of reviews
    pro_reviews_count=('sentiment', lambda x: sum((x >= 4) & (x <= 5))),  # Count reviews with sentiment 4-5
    con_reviews_count=('sentiment', lambda x: sum((x >= 1) & (x <= 2))),  # Count reviews with sentiment 1-2
    most_recent_review=('reviews.date', 'max')  # Get the date of the most recent review
).reset_index()

# Step 3: Create combined reviews for pro (4-5 sentiment) and con (1-2 sentiment)
grouped_data['product_review_context_combined_pro'] = grouped_data.apply(
    lambda row: combine_reviews_by_sentiment(
        data[(data['name'] == row['name']) & (data['category_cluster'] == row['category_cluster'])],
        sentiment_range=(4, 5),
        max_reviews=10  # Limit to 10 reviews
    ), axis=1
)

grouped_data['product_review_context_combined_con'] = grouped_data.apply(
    lambda row: combine_reviews_by_sentiment(
        data[(data['name'] == row['name']) & (data['category_cluster'] == row['category_cluster'])],
        sentiment_range=(1, 2),
        max_reviews=10  # Limit to 10 reviews
    ), axis=1
)

# Step 4: Extract top keywords for pro and con reviews
grouped_data['pro_keywords'] = grouped_data['product_review_context_combined_pro'].apply(
    lambda x: extract_top_keywords(x, n=5)
)

grouped_data['con_keywords'] = grouped_data['product_review_context_combined_con'].apply(
    lambda x: extract_top_keywords(x, n=5)
)

# Step 5: Calculate ranking for products with more than 100 reviews
grouped_data['top_product_ranking'] = (
    grouped_data[grouped_data['review_count'] > 100]
    .groupby('category_cluster')['mean_sentiment']
    .rank(ascending=False, method='dense')
)

# Step 6: Sort the data
sorted_data = grouped_data.sort_values(
    by=['category_cluster', 'top_product_ranking'],
    ascending=[True, True]
)

# Step 7: Select the required columns
final_data = sorted_data[
    ['cluster_name','category_cluster', 'name', 'mean_sentiment', 'review_count',
     'top_product_ranking', 'most_recent_review',
     'product_review_context_combined_pro', 'product_review_context_combined_con',
     'pro_reviews_count', 'con_reviews_count', 'pro_keywords', 'con_keywords']
].rename(columns={'name': 'product_name'})

# # Save and Inspect to CSV
# output_file = "product_reviews_enhanced.csv"
# final_data.to_csv(output_file, index=False)

# print(f"CSV file has been saved as {output_file}")


# In[6]:


# Load .env file if used
load_dotenv()

# Get the API key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# In[7]:


# Load your example CSV Output
example_data = pd.read_csv("blog_posts_for_website_example.csv")

example_data.columns


# In[8]:


final_data.rename(columns={'category_cluster': 'Product_Category'}, inplace=True)


# In[9]:


final_data.columns


# In[11]:


# changes reordering and all columns / reduce to just top 1 

# Load your dataset
example_data = pd.read_csv("blog_posts_for_website_example.csv")

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)  # Replace with your Cohere API Key

# Define a function to generate content using Cohere
def generate_content(prompt):
    response = co.generate(
        model='command-xlarge-nightly',  # Use Cohere's advanced model
        prompt=prompt,
        max_tokens=300,  # Adjust token limit based on the column
        temperature=0.7,  # For balanced creativity and coherence
        stop_sequences=["\n"]  # Stops generation after the first paragraph
    )
    return response.generations[0].text.strip()

# Define prompt templates for each column
def create_prompt_for_column(column_name, row, example):
    """
    Create a prompt for generating specific columns.
    :param column_name: The column to generate content for.
    :param row: The row from the dataset.
    :param example: An example row from the blog_posts_for_website_example.csv.
    :return: A string prompt.
    """
    # Handle missing data for review contexts
    positive_reviews = row['product_review_context_combined_pro'][:500] if row.get('product_review_context_combined_pro') else "No positive reviews available."
    negative_reviews = row['product_review_context_combined_con'][:500] if row.get('product_review_context_combined_con') else "No negative reviews available."

    if column_name == "Headline_for_TOP_Product":
        return f"""Here is an example headline for a top product:
        Product Name: {example['Product_Top_1']}
        Example Headline: {example['Headline_for_TOP_Product']}
        
        Now, create a headline for this product:
        Product Name: {row['product_name']}
        Category: {row['Product_Category']}
        Positive Sentiment: {row['pro_reviews_count']}
        Headline:"""

    if column_name == "Teaser_text":
        return f"""Here is an example teaser text for a product:
        Product Name: {example['Product_Top_1']}
        Example Teaser: {example['Teaser_text']}
        
        Now, create a teaser text for this product:
        Product Name: {row['product_name']}
        Category: {row['Product_Category']}
        Combined Reviews (Positive): {positive_reviews}
        Teaser:"""

    if column_name == "Product_Top_1_Pros":
        return f"""Here is an example summary of pros for a product:
        Product Name: {example['Product_Top_1']}
        Example Pros: {example['Product_Top_1_Pros']}
        
        Now, summarize the pros for this product based on customer reviews:
        Product Name: {row['product_name']}
        Positive Reviews: {positive_reviews}
        Based on customer reviews, list 2-3 positive technical bullet points of this product (each bullet point max 3 words). Focus on specific features and performance benefits mentioned by users. Write concise and professional bullet points:
        Pros:"""

    if column_name == "Product_Top_1_Cons":
        return f"""Here is an example summary of cons for a product:
        Product Name: {example['Product_Top_1']}
        Example Cons: {example['Product_Top_1_Cons']}
        
        Now, summarize the cons for this product based on customer reviews:
        Negative Reviews: {negative_reviews}
        Based on customer reviews, list 2-3 positive technical bullet points of this product (each bullet point max 3 words). Focus on specific features and performance concerns mentioned by users. Write concise and professional bullet points:

        Cons:"""

    if column_name == "Summary_Reviews_High":
        return f"""Here is an example summary of high reviews for a category:
        Category: {example['Product_Category']}
        Example Summary: {example['Summary_Reviews_High']}
        
        Now, summarize all high reviews for this category:
        Positive Reviews for All Products: {positive_reviews}
        Write a concise, professional summary (2-3 sentences) focusing on the main benefits and user feedback:
        No Category number or naming. 
        Summary:"""

    if column_name == "Summary_Reviews_Low":
        return f"""Here is an example summary of low reviews for a category:
        Category: {example['Product_Category']}
        Example Summary: {example['Summary_Reviews_Low']}
        
        Now, summarize all low reviews for this category:
        Negative Reviews for All Products: {negative_reviews}
        Write a concise, professional summary (2-3 sentences) focusing on the main issues and concerns raised by users:
        No Category number or naming. 
        Summary:"""

    if column_name == "Wrapup":
        return f"""Here is an example wrap-up for a category:
        Category: {example['Product_Category']}
        Example Wrapup: {example['Wrapup']}
        
        Now, write a wrap-up for this category:
        Highlights: {positive_reviews}
        Concerns: {negative_reviews}
        Write a professional wrap-up in 2-3 sentences that provides a balanced overview:
        No Category number or naming. 
        Wrapup:"""

# Filter and rank products within each category
final_data["rank"] = final_data.groupby("Product_Category")["top_product_ranking"].rank(ascending=False)
top_3_data = final_data[final_data["rank"] <= 3]  # Filter Top 3 products
top_product_data = top_3_data[top_3_data["rank"] == 1]  # Filter only the top-ranked product for certain fields


# Prepare the output DataFrame
output_columns = [
    "Product_Category", "Category_name", 
    "Headline_for_TOP_Product", "Teaser_text",
    "Product_Top_1", "Product_Top_1_Pro_Sentiment",
    "Product_Top_2", "Product_Top_2_Pro_Sentiment",
     "Product_Top_3", "Product_Top_3_Pro_Sentiment",
     "Product_Top_1_Pros", "Product_Top_1_Cons",
     "Positive Quote", "Summary_Reviews_High", "Summary_Reviews_Low", 
     "Wrapup"
]
output_data = []

# Iterate over the categories and combine Top 1, Top 2, and Top 3 into one row per category
categories = final_data["Product_Category"].unique()

for category in categories:
    category_data = final_data[final_data["Product_Category"] == category]  
    # Sort products by `top_product_ranking` (lower is better)
    category_data = category_data.sort_values(by="top_product_ranking")

    # Extract the top products
    product_top_1 = category_data.iloc[0]
    product_top_2 = category_data.iloc[1] if len(category_data) > 1 else None
    product_top_3 = category_data.iloc[2] if len(category_data) > 2 else None

    # Create a single row with all top products for the category
    row_output = {
        "Product_Category": product_top_1["Product_Category"],
        "Category_name": product_top_1["cluster_name"],
        "Headline_for_TOP_Product": generate_content(create_prompt_for_column("Headline_for_TOP_Product", product_top_1, example_data.iloc[0])),
        "Teaser_text": generate_content(create_prompt_for_column("Teaser_text", product_top_1, example_data.iloc[0])),
        "Product_Top_1": product_top_1["product_name"],
        "Product_Top_1_Pro_Sentiment": round(product_top_1["mean_sentiment"], 2),
        "Product_Top_1_Pros": generate_content(create_prompt_for_column("Product_Top_1_Pros", product_top_1, example_data.iloc[0])),
        "Product_Top_1_Cons": generate_content(create_prompt_for_column("Product_Top_1_Cons", product_top_1, example_data.iloc[0])),
        "Product_Top_2": product_top_2["product_name"] if product_top_2 is not None else None,
        "Product_Top_2_Pro_Sentiment": round(product_top_2["mean_sentiment"], 2) if product_top_2 is not None else None,
        "Product_Top_3": product_top_3["product_name"] if product_top_3 is not None else None,
        "Product_Top_3_Pro_Sentiment": round(product_top_3["mean_sentiment"], 2) if product_top_3 is not None else None,
        "Positive Quote": product_top_1["product_review_context_combined_pro"].split(".")[0] if product_top_1["product_review_context_combined_pro"] else "No positive quote available.",
        "Summary_Reviews_High": generate_content(create_prompt_for_column("Summary_Reviews_High", product_top_1, example_data.iloc[0])),
        "Summary_Reviews_Low": generate_content(create_prompt_for_column("Summary_Reviews_Low", product_top_1, example_data.iloc[0])),
        "Wrapup": generate_content(create_prompt_for_column("Wrapup", product_top_1, example_data.iloc[0])),
    }

    output_data.append(row_output)


# Convert the output to a DataFrame and save as CSV
output_df = pd.DataFrame(output_data, columns=output_columns)
output_df.to_csv("blog_posts_for_website.csv", index=False)

print("File 'blog_posts_for_website.csv' has been generated!")

