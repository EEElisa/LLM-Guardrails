import pandas as pd 
import json 

def prep_annotation_queries(df, n_queries, output_path, random_seed=42):
    """
    Sample n complete query sets (all 3 intents) from the dataframe
    
    Parameters:
    df: DataFrame with columns 'query_id', 'id', 'text', 'intent'
    n_queries: Number of queries to sample
    random_seed: Random seed for reproducibility
    
    Returns:
    DataFrame with all rows for the sampled queries
    """
    # Set random seed
    tdf = df[df['flag'] == True]
    sampled_tdf = tdf.groupby('category').sample(n=n_queries, random_state=random_seed)
    sampled_tdf = sampled_tdf.reset_index(drop=True)
    
    sampled_tdf_long = sampled_tdf.melt(
    id_vars=['category', 'topic', 'query'], 
    value_vars=['benign_scenario', 'malicious_scenario'], 
    var_name='scenario_type', 
    value_name='scenario'
    )

    # Map the 'intent' column based on the scenario type
    sampled_tdf_long['intent'] =      sampled_tdf_long['scenario_type'].map({
        'benign_scenario': 'benign',
        'malicious_scenario': 'malicious'
    })

    # Drop the unnecessary 'scenario_type' column
    sampled_tdf_long = sampled_tdf_long.drop(columns=['scenario_type'])
    sampled_tdf_long.to_csv(output_path, index=False)
    return sampled_tdf_long

def generate_task_id(df, category_abbreviations, intent_suffixes, output_path, query_index_start=101):
    queries_per_category = df.groupby('category')['query'].nunique()[0]
    df['query_index'] = df.groupby('category').cumcount() % queries_per_category + query_index_start
    df['category_abbr'] = df['category'].map(category_abbreviations)
    df['query_id'] = (df['category_abbr'] + "_" + df['query_index'].astype(str))
    df['id'] = (df['query_id'] + df['intent'].map(intent_suffixes))
    # Create the 'text' column integrating 'query' and 'scenario'
    df['text'] = df.apply(lambda row: f"<i>{row['scenario']}<i> <br> {row['query']}", axis=1)
    df.to_csv(output_path, index=False)
    return df

def filter_instances(annotated_instances):
    intents = []
    proficiencies = [] 
    
    for instance in annotated_instances:
        intent_dict = instance.get("label_annotations", {}).get("intent")
        intent_key = next(iter(intent_dict), None) if isinstance(intent_dict, dict) else None  # Get first key if intent_dict exists
        
        if intent_key is not None:  # Only store rows where intent exists
            intents.append({
                "user_id": instance["user_id"],
                "instance_id": instance["instance_id"],
                "displayed_text": instance["displayed_text"],
                "labeled_intent": intent_key
            })
            
        proficiency_rating = instance.get("label_annotations", {}).get("How would you rate your English proficiency?")
        proficiency_key = next(iter(proficiency_rating), None) if isinstance(proficiency_rating, dict) else None
        if proficiency_rating is not None:
            proficiencies.append({
                "user_id": instance["user_id"],
                "proficiency": proficiency_key
            })
        
    return intents, proficiencies

def preprocess_annotation_result(annotated_instances_file, truth_file, raw_file):
    with open(annotated_instances_file, "r", encoding="utf-8") as f:
        annotated_instances = [json.loads(line) for line in f if line.strip()]
    intents, proficiencies = filter_instances(annotated_instances)
    intent_df = pd.DataFrame(intents)
    proficiency_df = pd.DataFrame(proficiencies)
    annotation_result_df = pd.merge(intent_df, proficiency_df, on="user_id")
    true_intent_df = pd.read_csv(truth_file)
    annotation_result = pd.merge(annotation_result_df, true_intent_df[['id', 'text', 'query_id', 'intent']], left_on=['displayed_text', 'instance_id'], right_on=['text', 'id']).drop(columns=['text', 'id']).rename(columns={"intent": "true_intent"})
    user_counts = annotation_result.groupby('instance_id')['user_id'].nunique()
    valid_instances = user_counts[user_counts == 6].index
    filtered_result = annotation_result[annotation_result['instance_id'].isin(valid_instances)]
    
    filtered_result['category'] = filtered_result['query_id'].str[:2]
    filtered_result['accuracy'] = (filtered_result['labeled_intent'] == filtered_result['true_intent']).astype(int)
    instance_to_query = filtered_result[['instance_id', 'query_id']].drop_duplicates()
    
    raw_df = pd.read_csv(raw_file)
    raw_df = raw_df[raw_df['query_id'].isin(filtered_result['query_id'])]
    query_df = raw_df[['query_id', 'query', 'category', 'topic']].drop_duplicates()
    intent_counts = filtered_result.groupby(['instance_id', 'labeled_intent']).size().unstack(fill_value=0)
    return filtered_result, query_df, instance_to_query, intent_counts

def get_majority_vote(row, threshold):
    # Get the intent with the highest count
    max_count = row.max()
    if max_count >= threshold:  # Only consider it a majority if count >= threshold
        return row.idxmax()
    else:
        return "no_majority"
    
def get_quorum_valid_queries(annotation_result, intent_counts, instance_to_query, query_df, threshold):
    majority_votes = intent_counts.apply(lambda row: get_majority_vote(row, threshold), axis=1).reset_index()
    majority_votes.columns = ['instance_id', 'majority_vote']
    instance_true_intents = annotation_result[['instance_id', 'true_intent', 'category']].drop_duplicates()
    instance_comparison = pd.merge(instance_true_intents, majority_votes, on='instance_id')
    instance_comparison['accuracy'] = (instance_comparison['majority_vote'] ==      instance_comparison['true_intent']).astype(int)
    instances_with_majority = instance_comparison[instance_comparison['majority_vote'] != 'no_majority']
    instances_with_majority = pd.merge(instances_with_majority, instance_to_query, on='instance_id')
    instances_valid_flag = instances_with_majority.groupby(['query_id', 'category'])["accuracy"].apply(
        lambda x: x.sum()==2
    ).reset_index()
    valid_queries = instances_valid_flag[instances_valid_flag['accuracy'] == True].reset_index(drop=True)
    print(f"====Threshold: {threshold}====")
    print(f"Number of valid queries: {valid_queries.groupby('category').size()}")
    valid_query_ids = valid_queries['query_id'].unique()
    valid_instances = query_df[query_df['query_id'].isin(valid_query_ids)].reset_index(drop=True)
    return valid_instances