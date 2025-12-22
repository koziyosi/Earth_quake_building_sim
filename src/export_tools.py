"""
Export Utilities Module.
Various export formats for analysis results.
"""
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


def export_to_opensees(
    nodes: List,
    elements: List,
    output_path: str,
    analysis_type: str = 'dynamic'
) -> str:
    """
    Export model to OpenSees Tcl script.
    
    Args:
        nodes: List of Node objects
        elements: List of Element objects
        output_path: Output file path
        analysis_type: 'static', 'dynamic', or 'pushover'
        
    Returns:
        Path to created file
    """
    lines = [
        "# OpenSees Model exported from EarthQuake Building Sim",
        f"# Generated: {datetime.now().isoformat()}",
        "",
        "wipe",
        "model BasicBuilder -ndm 3 -ndf 6",
        "",
        "# ===== NODES ====="
    ]
    
    for i, node in enumerate(nodes):
        lines.append(f"node {i+1} {node.x:.4f} {node.y:.4f} {node.z:.4f}")
        
    lines.append("")
    lines.append("# ===== MATERIALS =====")
    lines.append("uniaxialMaterial Steel01 1 235e6 2.05e11 0.02")
    lines.append("uniaxialMaterial Concrete01 2 -30e6 -0.002 -20e6 -0.004")
    
    lines.append("")
    lines.append("# ===== SECTIONS =====")
    lines.append("section Elastic 1 2.05e11 0.01 1e-4 1e-4 8e10 1e-5")
    
    lines.append("")
    lines.append("# ===== GEOMETRIC TRANSFORMATIONS =====")
    lines.append("geomTransf Linear 1 0 0 1")
    lines.append("geomTransf PDelta 2 0 0 1")
    
    lines.append("")
    lines.append("# ===== ELEMENTS =====")
    
    for i, elem in enumerate(elements):
        node_i_idx = nodes.index(elem.node_i) + 1
        node_j_idx = nodes.index(elem.node_j) + 1
        
        # Determine element type
        dz = abs(elem.node_j.z - elem.node_i.z)
        L = elem.get_length()
        
        if L > 0 and dz / L > 0.9:  # Column
            lines.append(f"element elasticBeamColumn {i+1} {node_i_idx} {node_j_idx} 0.04 2.05e11 8e10 1e-4 1e-4 1e-5 2")
        else:  # Beam
            lines.append(f"element elasticBeamColumn {i+1} {node_i_idx} {node_j_idx} 0.03 2.05e11 8e10 5e-5 2e-4 1e-5 1")
            
    # Boundary conditions
    lines.append("")
    lines.append("# ===== BOUNDARY CONDITIONS =====")
    
    for i, node in enumerate(nodes):
        if node.z == 0:  # Base nodes
            lines.append(f"fix {i+1} 1 1 1 1 1 1")
            
    # Analysis setup
    lines.append("")
    lines.append("# ===== ANALYSIS =====")
    
    if analysis_type == 'dynamic':
        lines.extend([
            "",
            "# Mass assignment",
            "# mass nodeTag mx my mz Ixx Iyy Izz",
        ])
        for i, node in enumerate(nodes):
            if node.mass > 0:
                lines.append(f"mass {i+1} {node.mass} {node.mass} 0 0 0 0")
                
        lines.extend([
            "",
            "# Rayleigh damping",
            "rayleigh 0.5 0.001 0 0",
            "",
            "# Dynamic analysis",
            "constraints Plain",
            "numberer RCM",
            "system BandGeneral",
            "test NormDispIncr 1e-6 10",
            "algorithm Newton",
            "integrator Newmark 0.5 0.25",
            "analysis Transient",
        ])
        
    elif analysis_type == 'pushover':
        lines.extend([
            "",
            "# Pushover analysis",
            "constraints Plain",
            "numberer RCM", 
            "system BandGeneral",
            "test NormDispIncr 1e-6 10",
            "algorithm Newton",
            "integrator DisplacementControl 1 1 0.001",
            "analysis Static",
        ])
        
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
        
    return output_path


def export_to_latex(
    results: Dict[str, Any],
    output_path: str,
    title: str = "Seismic Analysis Report"
) -> str:
    """
    Export results to LaTeX document.
    
    Args:
        results: Analysis results dictionary
        output_path: Output file path (.tex)
        title: Document title
        
    Returns:
        Path to created file
    """
    lines = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{amsmath}",
        r"\usepackage[margin=2.5cm]{geometry}",
        "",
        r"\title{" + title + "}",
        r"\author{EarthQuake Building Sim}",
        r"\date{\today}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\section{Analysis Summary}",
    ]
    
    # Add summary table
    if 'summary' in results:
        lines.extend([
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"Parameter & Value \\",
            r"\midrule",
        ])
        
        for key, value in results['summary'].items():
            if isinstance(value, float):
                lines.append(f"{key} & {value:.4f} \\\\")
            else:
                lines.append(f"{key} & {value} \\\\")
                
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Analysis Summary}",
            r"\end{table}",
        ])
        
    # Response section
    if 'max_drift' in results:
        lines.extend([
            "",
            r"\section{Structural Response}",
            f"Maximum inter-story drift ratio: {results['max_drift']:.4f}",
            "",
        ])
        
    lines.extend([
        r"\end{document}",
    ])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
        
    return output_path


def export_to_excel_format(
    time: np.ndarray,
    data: Dict[str, np.ndarray],
    output_path: str
) -> str:
    """
    Export time history data to CSV format compatible with Excel.
    
    Args:
        time: Time array
        data: Dictionary of data arrays
        output_path: Output file path (.csv)
        
    Returns:
        Path to created file
    """
    headers = ['Time(s)'] + list(data.keys())
    
    with open(output_path, 'w', newline='') as f:
        f.write(','.join(headers) + '\n')
        
        for i in range(len(time)):
            row = [f'{time[i]:.4f}']
            for key in data.keys():
                if i < len(data[key]):
                    row.append(f'{data[key][i]:.6e}')
                else:
                    row.append('')
            f.write(','.join(row) + '\n')
            
    return output_path


def export_building_xml(
    nodes: List,
    elements: List,
    output_path: str,
    metadata: Dict = None
) -> str:
    """
    Export building model to XML format.
    
    Args:
        nodes: List of Node objects
        elements: List of Element objects
        output_path: Output file path (.xml)
        metadata: Optional metadata dictionary
        
    Returns:
        Path to created file
    """
    import xml.etree.ElementTree as ET
    
    root = ET.Element('BuildingModel')
    root.set('generator', 'EarthQuake Building Sim')
    root.set('version', '1.0')
    root.set('timestamp', datetime.now().isoformat())
    
    # Metadata
    if metadata:
        meta = ET.SubElement(root, 'Metadata')
        for key, value in metadata.items():
            item = ET.SubElement(meta, 'Item')
            item.set('key', str(key))
            item.text = str(value)
            
    # Nodes
    nodes_elem = ET.SubElement(root, 'Nodes')
    for i, node in enumerate(nodes):
        n = ET.SubElement(nodes_elem, 'Node')
        n.set('id', str(i))
        n.set('x', f'{node.x:.4f}')
        n.set('y', f'{node.y:.4f}')
        n.set('z', f'{node.z:.4f}')
        n.set('mass', f'{node.mass:.2f}')
        
    # Elements
    elements_elem = ET.SubElement(root, 'Elements')
    for i, elem in enumerate(elements):
        e = ET.SubElement(elements_elem, 'Element')
        e.set('id', str(i))
        e.set('node_i', str(nodes.index(elem.node_i)))
        e.set('node_j', str(nodes.index(elem.node_j)))
        e.set('type', type(elem).__name__)
        
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    return output_path


def export_animation_mp4(
    frames_path: str,
    output_path: str,
    fps: int = 30
) -> str:
    """
    Export animation frames to MP4 video.
    
    Uses ffmpeg if available, otherwise creates GIF.
    
    Args:
        frames_path: Path pattern to frames (e.g., 'frames/img_%04d.png')
        output_path: Output video path
        fps: Frames per second
        
    Returns:
        Path to created file
    """
    import subprocess
    import shutil
    
    # Check for ffmpeg
    if shutil.which('ffmpeg'):
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', frames_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    else:
        # Fallback: create GIF
        from PIL import Image
        import glob
        
        pattern = frames_path.replace('%04d', '*')
        frame_files = sorted(glob.glob(pattern))
        
        if not frame_files:
            raise FileNotFoundError(f"No frames found matching {pattern}")
            
        images = [Image.open(f) for f in frame_files]
        gif_path = output_path.replace('.mp4', '.gif')
        
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000/fps),
            loop=0
        )
        return gif_path


def save_to_sqlite(
    db_path: str,
    table_name: str,
    data: Dict[str, np.ndarray]
) -> str:
    """
    Save analysis results to SQLite database.
    
    Args:
        db_path: Database file path
        table_name: Table name
        data: Dictionary of arrays to save
        
    Returns:
        Database path
    """
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    columns = list(data.keys())
    col_defs = ', '.join([f'"{c}" REAL' for c in columns])
    cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" (id INTEGER PRIMARY KEY, {col_defs})')
    
    # Insert data
    n_rows = len(list(data.values())[0])
    placeholders = ', '.join(['?' for _ in columns])
    
    for i in range(n_rows):
        values = [float(data[c][i]) if i < len(data[c]) else None for c in columns]
        cursor.execute(f'INSERT INTO "{table_name}" ({", ".join(columns)}) VALUES ({placeholders})', values)
        
    conn.commit()
    conn.close()
    
    return db_path
