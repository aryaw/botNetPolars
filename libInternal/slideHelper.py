import os

def generate_algo_slide(algo_name, algo_dir):
    html = f"""
    <html>
    <head><title>{algo_name} Result</title></head>
    <body>
        <h1>{algo_name} – Botnet & C&C Detection</h1>
        <h2>Evaluation</h2>
        <iframe src="evaluation_table.html" width="100%" height="300"></iframe>
        <iframe src="confusion_matrix.html" width="100%" height="350"></iframe>
        <h2>C&C Candidates</h2>
        <iframe src="cnc_candidates.html" width="100%" height="300"></iframe>
        <h2>C&C Communication Graph</h2>
        <iframe src="graph_cnc.html" width="100%" height="600"></iframe>
    </body>
    </html>
    """
    with open(os.path.join(algo_dir, "slide.html"), "w") as f:
        f.write(html)
