import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Print column names to verify
    print("Columns in the DataFrame:", df.columns.tolist())
    return df

def find_two_level_neighbors(df, account_id):
    # Find first level neighbors where the account is either the sender or receiver
    # Note: When pandas loads a CSV with duplicate column names, it appends suffixes
    # The columns are likely named 'Account' and 'Account.1' or similar
    first_level = df[(df['Account'] == account_id) | (df['Account.1'] == account_id)]
    
    # Collect all unique accounts that are first-level neighbors
    source_accounts = set(first_level['Account'])
    target_accounts = set(first_level['Account.1'])
    first_level_accounts = source_accounts.union(target_accounts)
    
    # Find second level neighbors
    second_level = df[(df['Account'].isin(first_level_accounts)) | (df['Account.1'].isin(first_level_accounts))]
    second_level_source = set(second_level['Account'])
    second_level_target = set(second_level['Account.1'])
    second_level_accounts = second_level_source.union(second_level_target)
    
    # Combine first and second level neighbors
    all_neighbors = first_level_accounts.union(second_level_accounts)
    subset_df = df[(df['Account'].isin(all_neighbors)) | (df['Account.1'].isin(all_neighbors))]
    
    return subset_df

def visualize_network(subset_df):
    # Create a directed graph from the dataframe
    G = nx.DiGraph()
    
    # Add edges from the dataframe
    for _, row in subset_df.iterrows():
        G.add_edge(row['Account'], row['Account.1'])
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    plt.title("Two-Level Deep Neighbors Network")
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = './data/IBM_AML/HI-Small_Trans.csv'
    account_id = '80100D180'  # Replace with the actual account id
    df = load_data(file_path)
    subset_df = find_two_level_neighbors(df, account_id)
    visualize_network(subset_df)
