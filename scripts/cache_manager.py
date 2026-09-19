
import os
import sys

# Running `python scripts/<name>.py` puts scripts/ on sys.path, not the
# repository root, so the queryagent package would not be importable. Add the
# root explicitly rather than requiring `python -m scripts.<name>`, which is
# not what anyone types.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



#!/usr/bin/env python3
"""
Cache management tool for ChromaDB cache
"""

import argparse
from queryagent.cache import (
    view_all_cache, search_cache, delete_cache_item, 
    delete_cache_by_query, clear_cache, get_cache_stats
)

def main():
    parser = argparse.ArgumentParser(description='ChromaDB Cache Manager')
    parser.add_argument('action', choices=['view', 'search', 'delete', 'clear', 'stats'], 
                       help='Action to perform')
    parser.add_argument('--query', '-q', help='Query text for search/delete')
    parser.add_argument('--id', help='Item ID for delete')
    parser.add_argument('--limit', '-l', type=int, default=10, help='Limit results')
    
    args = parser.parse_args()
    
    if args.action == 'view':
        print("📋 Viewing all cached queries:")
        items = view_all_cache()
        for i, item in enumerate(items[:args.limit]):
            print(f"\n{i+1}. ID: {item['id']}")
            print(f"   Query: {item['query']}")
            print(f"   Summary: {item['summary']}")
        
        if len(items) > args.limit:
            print(f"\n... and {len(items) - args.limit} more items")
        print(f"\nTotal items: {len(items)}")
    
    elif args.action == 'search':
        if not args.query:
            print("❌ Please provide search query with --query")
            return
        
        print(f"🔍 Searching for: {args.query}")
        items = search_cache(args.query)
        
        if not items:
            print("No matching items found")
        else:
            for i, item in enumerate(items):
                print(f"\n{i+1}. ID: {item['id']}")
                print(f"   Query: {item['query']}")
                print(f"   Summary: {item['summary']}")
    
    elif args.action == 'delete':
        if args.id:
            print(f"🗑️ Deleting item with ID: {args.id}")
            success = delete_cache_item(args.id)
            if success:
                print("✅ Item deleted successfully")
            else:
                print("❌ Failed to delete item")
        elif args.query:
            print(f"🗑️ Deleting items matching query: {args.query}")
            deleted_count = delete_cache_by_query(args.query)
            print(f"✅ Deleted {deleted_count} items")
        else:
            print("❌ Please provide --id or --query for deletion")
    
    elif args.action == 'clear':
        confirm = input("⚠️ Are you sure you want to clear entire cache? (y/N): ")
        if confirm.lower() == 'y':
            clear_cache()
            print("✅ Cache cleared successfully")
        else:
            print("❌ Operation cancelled")
    
    elif args.action == 'stats':
        print("📊 Cache Statistics:")
        stats = get_cache_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

if __name__ == "__main__":
    main()

'''
# View all cached queries (first 10)
python cache_manager.py view

# View more items
python cache_manager.py view --limit 20

# Search for specific queries
python cache_manager.py search --query "python"

# Delete by query text
python cache_manager.py delete --query "python"

# Delete by ID
python cache_manager.py delete --id "abc123-def456"

# Clear entire cache
python cache_manager.py clear

# Get cache statistics
python cache_manager.py stats
'''