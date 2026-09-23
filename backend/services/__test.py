patient_details = {
    "P001": {"name": "Alice", "height": 165, "weight": 60},
    "P002": {"name": "Bob", "height": 180, "weight": 75},
    "P003": {"name": "Charlie", "height": 170, "weight": 68}
}

# Sort by height in ascending order
sorted_by_height = sorted(
    patient_details.values(), 
    key=lambda item: item["height"],
    reverse=False
)
num = [1,2,3,4,5,6,7]
print(sorted_by_height)
print('[**num]')
# dict_items([('P001', {'name': 'Alice', 'height': 165, 'weight': 60}), ('P002', {'name': 'Bob', 'height': 180, 'weight': 75}), ('P003', {'name': 'Charlie', 'height': 170, 'weight': 68})])
# dict_values([{'name': 'Alice', 'height': 165, 'weight': 60}, {'name': 'Bob', 'height': 180, 'weight': 75}, {'name': 'Charlie', 'height': 170, 'weight': 68}])

