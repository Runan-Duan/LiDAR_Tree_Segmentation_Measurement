import subprocess

# Convert LAZ to PCD using PDAL
def convert_laz_to_pcd(input_file, output_file):
    command = f"pdal translate {input_file} {output_file}"
    subprocess.run(command, shell=True)

# Example usage
convert_laz_to_pcd('input.laz', 'output.pcd')
