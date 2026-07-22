from itertools import islice

name, axis = input("Enter the name your file including the extension and the location if not in current directory (ex. input.xyz): "), input("Enter the name of the axis (x, y, or z): ")
xyz_data = []


with open(name, "r") as f:
   atom_count_line = f.readline().strip()
   comment_line = f.readline().strip()
   for line in f:
       line2=line.strip()
       parts = line2.split()
       xyz_data.append([parts[0], float(parts[1]), float(parts[2]), float(parts[3])])

sorted_data = sorted(xyz_data, key=lambda x: (x[{"x": 1, "y": 2, "z": 3}[axis]]))

clean_name = name.removesuffix(".xyz")
name = clean_name + "_" + axis + "sorted.xyz"

with open(name, "w") as f:
    f.write(atom_count_line+"\n")
    f.write(comment_line+" Sorted by " + axis + "\n")
    for item in sorted_data:
        f.write(f"{item[0]} {item[1]} {item[2]} {item[3]}\n")