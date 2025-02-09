import pandas as pd
import numpy as np
import networkx as nx
import mygene
import random
import ranky as rky

# Returns a dataframe with genename_cpgname ('combined') as index, a column with the score created by GRNBoost2, a 'genes' and a 'scores column
def prepare_grnboost_output(i):
    file_i = pd.read_csv('output/grn_output_' + str(i) +'.tsv', sep='\t')
    column_names = ["genes", "cpgs", "score_"+str(i)]
    file_i.columns = column_names
    
    # Create new column with genes_cpgs
    file_i['combined'] = file_i['genes'].astype(str) + '_' + file_i['cpgs'].astype(str)
    file_i = file_i.set_index('combined')
    
    if i>1:
        file_i = file_i.drop(['genes', 'cpgs'], axis=1)
    return file_i

def create_aggregated_ranking():
    # Concatenate the first 10 outputs by GRNBoost2
    file_1 = prepare_grnboost_output(1)
    file_2 = prepare_grnboost_output(2)
    combined = pd.concat([file_1, file_2], axis=1)
    nruns = read_nruns_from_file()
    for i in range(3,nruns+1):
        file_i = prepare_grnboost_output(i)
        combined = pd.concat([combined, file_i], axis=1)

    # Sort by "combined"
    combined = combined.sort_values(by=['combined'], axis = 0) 

    # Fill NaN values with 0
    combined = combined.fillna(0)

    # Transform scores to ranking values
    for i in range(1,nruns+1):
        combined[['ranking_' + str(i)]] = combined[['score_' + str(i)]].rank(method='dense', ascending=False).astype(int)
        combined = combined.drop(columns=['score_' + str(i)])

    combined = combined.drop(columns=['cpgs'])
    combined.to_csv('combined.tsv', sep='\t', index=False)
    combined.reset_index(drop=True, inplace=True)  # Remove the current index
    combined.set_index(combined.columns[0], inplace=True)

    # Aggregate rankings
    aggr_ranking = rky.borda(combined, axis=1)
    aggr_ranking = aggr_ranking.sort_values(ascending=False)

    return aggr_ranking

# Load HIPPIE network and return a NetworkX graph
def load_hippie_network(file_path):
    hippie = pd.read_csv(file_path, sep='\t', usecols=[1, 3], names=['gene1', 'gene2'])
    hippie_nx = nx.Graph()  
    hippie_nx.add_edges_from(zip(hippie['gene1'], hippie['gene2'])) 
    return hippie_nx, hippie

def get_gene_info(genes):
    mg = mygene.MyGeneInfo()  
    gene_info = mg.getgenes(list(genes), fields='symbol')
    return gene_info

def map_symbols_to_entrez(gene_info):
    return {int(d['_id']): d['symbol'] for d in gene_info if 'symbol' in d}

# Return a dictionary of neighboring genes for each gene 'genes'
def get_neighbors(graph, genes):
    return {gene: list(graph.neighbors(gene)) for gene in genes}

def calculate_degrees(neighbors):
    return {gene: len(neigh) for gene, neigh in neighbors.items()}

# Return a dictionary mapping each gene to a list of genes with the same degree in the hippie network
def get_same_degree_genes(degrees, graph_degrees):
    return {gene: [k for k, v in graph_degrees.items() if v == degree] for gene, degree in degrees.items()}

# Calculates p-values by comparing the mean PageRank of TFs with the mean PageRank of random gene sets, using n_iter iterations
def calculate_p_values(hippie_nx, tfs_degrees, same_node_degree, dmts_symbol2entrez, intersec_symbol2entrez, n_iter=30, random_iter=1000):
    # n_iter: number of iterations to run PageRank algorithm and calculate mean PageRank
    # random_iter: number of random gene sets to generate for comparison
    
    p_values = []
    
    for _ in range(n_iter):
        # Generate random sets of genes by randomly selecting genes with the same degree as TFs
        random_sets = [[random.choice(same_node_degree[tf]) for tf in tfs_degrees] for _ in range(random_iter)]
        
        # Initialize personalization vector for PageRank using the DNMTS and TETs
        person = {v: 1 / len(dmts_symbol2entrez) for v in dmts_symbol2entrez.values()}
        
        # Compute the PageRank of the network with personalized vector
        pr = nx.pagerank(hippie_nx, personalization=person)
        
        # Calculate mean PageRank for the intersection of TFs and DMTS genes
        mean_tf = np.mean([pr[k] for k in intersec_symbol2entrez.values()])
        
        # Calculate mean PageRank for each random gene set
        mean_random = [
            np.mean([pr[k] for k in random_set]) for random_set in random_sets
        ]
        
        # Calculate p-value by comparing the mean PageRank of the random sets to the mean PageRank of the TFs
        p_value = (np.sum(np.array(mean_random) >= mean_tf) + 1) / (len(mean_random) + 1)
        
        p_values.append(p_value) 
    
    return p_values

# Read nruns from file
def read_nruns_from_file():
    with open('output/nruns.txt', 'r') as file:
        nruns = int(file.read().strip())
    return nruns

def main():
    aggr_ranking = create_aggregated_ranking()
    ensID_TFs = list(dict.fromkeys(aggr_ranking.index)) # Remove duplicates, keep order
    ensID_top30TFs = ensID_TFs[:30] 
    
    # Load HIPPIE network and extract gene information
    hippie_nx, hippie = load_hippie_network('utils/hippie.txt')
    hippie_all_genes = set(hippie['gene1']).union(set(hippie['gene2']))  # Get all genes from the network
    hippie_gene_info = get_gene_info(hippie_all_genes)  # Fetch gene info
    
    # Map Entrez gene IDs to gene symbols for the HIPPIE network
    hippie_entrez2symbol = map_symbols_to_entrez(hippie_gene_info)
    
    # Get gene symbols for the top 30 TFs
    tfs_gene_info = get_gene_info(ensID_top30TFs)
    tfs_ensemble2symbol = {d['query']: d['symbol'] for d in tfs_gene_info if 'symbol' in d}
    
    # Find the intersection of TFs and HIPPIE genes
    intersec_genes = [x for x in tfs_ensemble2symbol.values() if x in hippie_entrez2symbol.values()]
    intersec_symbol2entrez = {v: k for k, v in hippie_entrez2symbol.items() if v in intersec_genes}
    
    # Define DNMTs and TETs and their Entrez IDs
    dmts_symbol2entrez = {'DNMT1': 1786, 'DNMT3A': 1788, 'DNMT3B': 1789, 'TET1': 80312, 'TET2': 54790, 'TET3': 200424}

    # Get neighbors and degrees of the TFs in the network
    tfs_neighbors = get_neighbors(hippie_nx, intersec_symbol2entrez.values())
    tfs_degrees = calculate_degrees(tfs_neighbors)
    hippie_degrees = dict(hippie_nx.degree())

    # Get genes with same degree as the TFs in the network
    same_node_degree = get_same_degree_genes(tfs_degrees, hippie_degrees)
    
    # Calculate p-values based on PageRank comparisons
    p_values = calculate_p_values(hippie_nx, tfs_degrees, same_node_degree, dmts_symbol2entrez, intersec_symbol2entrez)
    
    # Output the mean p-value
    print("The mean p-value is", np.mean(p_values))

if __name__ == "__main__":
    main()