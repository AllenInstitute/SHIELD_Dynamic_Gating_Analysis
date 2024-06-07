import numpy as np
import pandas as pd


def get_parent(area, structure_tree):
    
    parent_id = structure_tree[structure_tree['acronym']==area]['parent_structure_id']#.values[0]
    if len(parent_id)==0:
        return pd.DataFrame({'name': 'area not in structure tree', 
                             'parent_structure_id': np.nan,
                             'acronym': 'noarea'}, index=[0])
    return structure_tree[structure_tree['id'] == parent_id.values[0]]
    

def get_area_name_from_acronym(acronym, structure_tree):
    
    area = structure_tree[structure_tree['acronym']==acronym]
    if len(area)==0:
        return 'area not in structure tree'
    
    return area['name'].values[0]
    

def list_parents(area, structure_tree):
    if area == 'root':
        return []
    
    area_name = get_area_name_from_acronym(area, structure_tree) 
    parent_list = [area_name]
    parent_structure_id = 0.0
    while not np.isnan(parent_structure_id):
        parent = get_parent(area, structure_tree)
        parent_list.append(parent['name'].values[0])
        parent_structure_id = parent['parent_structure_id'].values[0]
        area = parent['acronym'].values[0]
    
    return parent_list


def get_brain_division_for_area(area, structure_tree, 
                                divisions = ['Isocortex', 'Hippocampal formation', 
                                             'Thalamus', 'Midbrain', 'Hypothalamus',
                                             'Striatum'],
                               cached_dict = None):
    
    if cached_dict:
        if area in cached_dict.keys():
            return cached_dict[area]
        
    parents = list_parents(area, structure_tree)
    intersection = np.intersect1d(parents, divisions)
    if len(intersection)>0:
        return intersection[0]
    
    return 'not in list'


def get_area_color(area, structure_tree):
    
    color = structure_tree[structure_tree['acronym']==area]['color_hex_triplet']
    if len(color)>0:
        color = color.values[0]
    else:
        color = '000000'

    return '#' + color


def add_brain_division_to_units_table(units_df, structure_tree):

    units_df['brain_division'] = units_df.apply(lambda row: get_brain_division_for_area(row['structure_acronym'], structure_tree), axis=1)

    return units_df