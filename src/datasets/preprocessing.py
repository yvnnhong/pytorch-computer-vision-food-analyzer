import torch
from torchvision.datasets import Food101
from torchvision import transforms
import os
import json
from pathlib import Path

def download_food101_dataset():
    """Download Food-101 dataset using torchvision"""
    print("Downloading Food-101 dataset...")
    
    # downloads to data/raw folder
    dataset = Food101(root='./data/raw', download=True, split='train')
    test_dataset = Food101(root='./data/raw', download=True, split='test')
    
    print(f"Training samples: {len(dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("Download complete!")
    
    return dataset, test_dataset

def create_cuisine_mappings():
    """Create food-to-cuisine mapping for all 101 Food-101 classes"""
    
    # Food-101 class names to cuisine mapping
    #hardcoded since food101 doesnt contain the cuisine type. 
    cuisine_mapping = {
        # Italian
        'pizza': 'Italian',
        'spaghetti_carbonara': 'Italian', 
        'spaghetti_bolognese': 'Italian',
        'lasagna': 'Italian',
        'ravioli': 'Italian',
        'risotto': 'Italian',
        'gnocchi': 'Italian',
        'caprese_salad': 'Italian',
        'bruschetta': 'Italian',
        'tiramisu': 'Italian',
        
        # Asian
        'sushi': 'Asian',
        'ramen': 'Asian',
        'fried_rice': 'Asian',
        'dumplings': 'Asian',
        'spring_rolls': 'Asian',
        'pad_thai': 'Asian',
        'miso_soup': 'Asian',
        'sashimi': 'Asian',
        'tempura': 'Asian',
        'bibimbap': 'Asian',
        'pho': 'Asian',
        'takoyaki': 'Asian',
        'gyoza': 'Asian',
        'peking_duck': 'Asian',
        'waffles': 'Asian',  # Note: some might be fusion
        
        # American
        'hamburger': 'American',
        'hot_dog': 'American',
        'french_fries': 'American',
        'apple_pie': 'American',
        'donuts': 'American',
        'pancakes': 'American',
        'chicken_wings': 'American',
        'mac_and_cheese': 'American',
        'grilled_cheese_sandwich': 'American',
        'club_sandwich': 'American',
        'cheesecake': 'American',
        'chocolate_cake': 'American',
        'ice_cream': 'American',
        'strawberry_shortcake': 'American',
        'red_velvet_cake': 'American',
        'baby_back_ribs': 'American',
        'pulled_pork_sandwich': 'American',
        'chicken_quesadilla': 'American',
        'caesar_salad': 'American',
        'cobb_salad': 'American',
        
        # Mexican
        'tacos': 'Mexican',
        'burritos': 'Mexican',
        'nachos': 'Mexican',
        'quesadilla': 'Mexican',
        'guacamole': 'Mexican',
        'churros': 'Mexican',
        'fajitas': 'Mexican',
        'enchiladas': 'Mexican',
        'tamales': 'Mexican',
        'ceviche': 'Mexican',
        
        # Mediterranean/Middle Eastern
        'hummus': 'Mediterranean',
        'falafel': 'Mediterranean',
        'greek_salad': 'Mediterranean',
        'baklava': 'Mediterranean',
        'tabbouleh': 'Mediterranean',
        'shawarma': 'Middle Eastern',
        'kebab': 'Middle Eastern',
        
        # Indian
        'samosa': 'Indian',
        'biryani': 'Indian',
        'naan_bread': 'Indian',
        'chicken_curry': 'Indian',
        'butter_chicken': 'Indian',
        'dal': 'Indian',
        
        # French
        'french_toast': 'French',
        'croissant': 'French',
        'quiche': 'French',
        'crepes': 'French',
        'macarons': 'French',
        'french_onion_soup': 'French',
        'beef_bourguignon': 'French',
        'ratatouille': 'French',
        'coq_au_vin': 'French',
        'croque_monsieur': 'French',
        'escargots': 'French',
        'fois_gras': 'French',
        'bouillabaisse': 'French',
        
        # Other/Fusion
        'fish_and_chips': 'British',
        'shepherd_pie': 'British',
        'beef_wellington': 'British',
        'bangers_and_mash': 'British',
        'paella': 'Spanish',
        'gazpacho': 'Spanish',
        'tapas': 'Spanish',
        'wiener_schnitzel': 'German',
        'sauerbraten': 'German',
        'pretzel': 'German',
        'sauerkraut': 'German',
        'goulash': 'Eastern European',
        'pierogies': 'Eastern European',
        'borscht': 'Eastern European',
        
        # Breakfast/General
        'eggs_benedict': 'American',
        'omelette': 'French',
        'breakfast_burrito': 'Mexican',
        'bagel': 'American',
        'muffin': 'American',
        'scone': 'British',
        
        # Desserts (assign based on origin)
        'carrot_cake': 'American',
        'panna_cotta': 'Italian',
        'cannoli': 'Italian',
        'gelato': 'Italian',
        'sorbet': 'French',
        'mousse': 'French',
        'profiteroles': 'French',
        'eclairs': 'French',
        'bread_pudding': 'British',
        'trifle': 'British',
        
        # Soups
        'tom_yum_soup': 'Asian',
        'chicken_noodle_soup': 'American',
        'clam_chowder': 'American',
        'minestrone': 'Italian',
        'potato_soup': 'American',
        
        # Salads
        'waldorf_salad': 'American',
        'potato_salad': 'American',
        'coleslaw': 'American',
        'garden_salad': 'American',
        
        # Additional items (default to fusion if unclear)
        'sandwich': 'Fusion',
        'soup': 'Fusion',
        'salad': 'Fusion',
        'wrap': 'Fusion',
        'smoothie': 'Fusion',
        'juice': 'Fusion'
    }
    
    # Create nutrition database dict with temp values 
    nutrition_db = {}
    for food in cuisine_mapping.keys():
        # Placeholder nutrition values til I get real nutrition data
        nutrition_db[food] = {
            'calories': 250 + hash(food) % 300,  # 250-550 range
            'protein': 10 + hash(food) % 20,     # 10-30g range  
            'carbs': 20 + hash(food) % 40,       # 20-60g range
            'fat': 5 + hash(food) % 25           # 5-30g range
        }
    
    # Ensure data directory exists
    os.makedirs('./data', exist_ok=True)
    
    # Save mappings
    with open('./data/cuisine_mappings.json', 'w') as f:
        json.dump(cuisine_mapping, f, indent=2)
    
    with open('./data/nutrition_db.json', 'w') as f:
        json.dump(nutrition_db, f, indent=2)
    
    print(f"Created cuisine mappings for {len(cuisine_mapping)} foods")
    print(f"Created nutrition database for {len(nutrition_db)} foods")
    
    return cuisine_mapping, nutrition_db

def setup_data_directories():
    """Create necessary data directories"""
    directories = [
        './data/raw',
        './data/processed', 
        './models',
        './logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    print("Setting up Food Analysis Project Data...")
    setup_data_directories()
    print("\nCreating food mappings...")
    create_cuisine_mappings()
    print("\nDownloading Food-101 dataset...")
    download_food101_dataset()
    print("\nData setup complete! Ready to start training.")