import os

output_dir = "lime_outputs"
html_index_path = os.path.join(output_dir, "index.html")

# Get all explanation files
explanation_files = sorted(
    [f for f in os.listdir(output_dir) if f.startswith("lime_explanation_") and f.endswith(".html")]
)

# Write index.html
with open(html_index_path, "w") as f:
    f.write("<html><head><title>LIME Explanations</title></head><body>\n")
    f.write("<h2>Resume Explanations</h2><ul>\n")
    for filename in explanation_files:
        f.write(f'<li><a href="{filename}" target="_blank">{filename}</a></li>\n')
    f.write("</ul></body></html>")

print(f"✅ Generated {html_index_path}")
