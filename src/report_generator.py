"""
Report Generator Module.
Generates analysis reports in various formats (#54).
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict
import numpy as np


class ReportGenerator:
    """
    Generates analysis reports in HTML, Markdown, or JSON formats.
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.data: Dict[str, Any] = {
            'metadata': {},
            'model': {},
            'input': {},
            'results': {},
            'figures': []
        }
        
    def set_metadata(
        self,
        title: str,
        project_name: str = "",
        engineer: str = "",
        date: str = None
    ):
        """Set report metadata."""
        self.data['metadata'] = {
            'title': title,
            'project_name': project_name,
            'engineer': engineer,
            'date': date or datetime.now().strftime("%Y-%m-%d %H:%M"),
            'generated_by': 'Earthquake Building Simulator'
        }
        
    def set_model_info(
        self,
        n_stories: int,
        n_nodes: int,
        n_elements: int,
        total_mass: float,
        total_height: float,
        structure_type: str = "Steel Frame"
    ):
        """Set model information."""
        self.data['model'] = {
            'n_stories': n_stories,
            'n_nodes': n_nodes,
            'n_elements': n_elements,
            'total_mass': total_mass,
            'total_height': total_height,
            'structure_type': structure_type
        }
        
    def set_input_motion(
        self,
        name: str,
        pga: float,
        duration: float,
        dt: float,
        source: str = ""
    ):
        """Set input motion information."""
        self.data['input'] = {
            'name': name,
            'pga': pga,
            'duration': duration,
            'dt': dt,
            'source': source
        }
        
    def set_results(
        self,
        max_drift: float,
        max_disp: float,
        max_accel: float,
        base_shear_coef: float,
        n_yielded: int,
        story_responses: List[Dict] = None
    ):
        """Set analysis results."""
        self.data['results'] = {
            'max_drift': max_drift,
            'max_disp': max_disp,
            'max_accel': max_accel,
            'base_shear_coef': base_shear_coef,
            'n_yielded_elements': n_yielded,
            'story_responses': story_responses or []
        }
        
    def add_figure(self, filepath: str, caption: str):
        """Add a figure to the report."""
        self.data['figures'].append({
            'path': filepath,
            'caption': caption
        })
        
    def generate_html(self, filename: str = "report.html") -> str:
        """Generate HTML report."""
        filepath = os.path.join(self.output_dir, filename)
        
        html = self._generate_html_content()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
            
        return filepath
    
    def generate_markdown(self, filename: str = "report.md") -> str:
        """Generate Markdown report."""
        filepath = os.path.join(self.output_dir, filename)
        
        md = self._generate_markdown_content()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md)
            
        return filepath
    
    def generate_json(self, filename: str = "report.json") -> str:
        """Generate JSON report."""
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert any numpy arrays to lists
        data = self._serialize_data(self.data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        return filepath
    
    def _serialize_data(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._serialize_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_data(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    def _generate_html_content(self) -> str:
        """Generate HTML content."""
        meta = self.data['metadata']
        model = self.data['model']
        inp = self.data['input']
        results = self.data['results']
        
        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{meta.get('title', 'Analysis Report')}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .report {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #2196F3;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2196F3;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background: #2196F3;
            color: white;
        }}
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        .result-box {{
            display: inline-block;
            padding: 15px 25px;
            margin: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            text-align: center;
        }}
        .result-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .result-label {{
            font-size: 12px;
            opacity: 0.8;
        }}
        .figure {{
            margin: 20px 0;
            text-align: center;
        }}
        .figure img {{
            max-width: 100%;
            border-radius: 5px;
        }}
        .figure-caption {{
            color: #666;
            font-style: italic;
            margin-top: 5px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="report">
        <h1>📊 {meta.get('title', 'Seismic Analysis Report')}</h1>
        
        <p><strong>Project:</strong> {meta.get('project_name', '-')}</p>
        <p><strong>Engineer:</strong> {meta.get('engineer', '-')}</p>
        <p><strong>Date:</strong> {meta.get('date', '-')}</p>
        
        <h2>🏢 Model Information</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Structure Type</td><td>{model.get('structure_type', '-')}</td></tr>
            <tr><td>Number of Stories</td><td>{model.get('n_stories', '-')}</td></tr>
            <tr><td>Total Height</td><td>{model.get('total_height', 0):.2f} m</td></tr>
            <tr><td>Number of Nodes</td><td>{model.get('n_nodes', '-')}</td></tr>
            <tr><td>Number of Elements</td><td>{model.get('n_elements', '-')}</td></tr>
            <tr><td>Total Mass</td><td>{model.get('total_mass', 0)/1000:.1f} ton</td></tr>
        </table>
        
        <h2>🌊 Input Motion</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Record Name</td><td>{inp.get('name', '-')}</td></tr>
            <tr><td>Peak Ground Acceleration</td><td>{inp.get('pga', 0)*100:.0f} gal ({inp.get('pga', 0):.2f} m/s²)</td></tr>
            <tr><td>Duration</td><td>{inp.get('duration', 0):.1f} s</td></tr>
            <tr><td>Time Step</td><td>{inp.get('dt', 0)*1000:.1f} ms</td></tr>
        </table>
        
        <h2>📈 Key Results</h2>
        <div style="text-align: center;">
            <div class="result-box">
                <div class="result-value">{results.get('max_drift', 0)*100:.3f}%</div>
                <div class="result-label">Max Inter-story Drift</div>
            </div>
            <div class="result-box">
                <div class="result-value">{results.get('max_disp', 0)*100:.1f} cm</div>
                <div class="result-label">Max Displacement</div>
            </div>
            <div class="result-box">
                <div class="result-value">{results.get('max_accel', 0)/9.81:.2f} g</div>
                <div class="result-label">Max Acceleration</div>
            </div>
            <div class="result-box">
                <div class="result-value">{results.get('base_shear_coef', 0):.3f}</div>
                <div class="result-label">Base Shear Coef.</div>
            </div>
        </div>
        
        <p><strong>Yielded Elements:</strong> {results.get('n_yielded_elements', 0)}</p>
"""
        
        # Story responses table
        if results.get('story_responses'):
            html += """
        <h2>📊 Story-by-Story Results</h2>
        <table>
            <tr><th>Story</th><th>Max Drift (%)</th><th>Max Disp (cm)</th><th>Ductility</th></tr>
"""
            for sr in results['story_responses']:
                html += f"""            <tr>
                <td>{sr.get('story', '-')}</td>
                <td>{sr.get('max_drift', 0)*100:.3f}</td>
                <td>{sr.get('max_disp', 0)*100:.2f}</td>
                <td>{sr.get('ductility', 0):.2f}</td>
            </tr>
"""
            html += "        </table>\n"
            
        # Figures
        if self.data['figures']:
            html += "        <h2>📷 Figures</h2>\n"
            for fig in self.data['figures']:
                html += f"""        <div class="figure">
            <img src="{fig['path']}" alt="{fig['caption']}">
            <div class="figure-caption">{fig['caption']}</div>
        </div>
"""
        
        html += f"""
        <div class="footer">
            <p>Generated by {meta.get('generated_by', 'Earthquake Building Simulator')}</p>
            <p>{meta.get('date', '')}</p>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_markdown_content(self) -> str:
        """Generate Markdown content."""
        meta = self.data['metadata']
        model = self.data['model']
        inp = self.data['input']
        results = self.data['results']
        
        md = f"""# {meta.get('title', 'Seismic Analysis Report')}

**Project:** {meta.get('project_name', '-')}  
**Engineer:** {meta.get('engineer', '-')}  
**Date:** {meta.get('date', '-')}

---

## Model Information

| Parameter | Value |
|-----------|-------|
| Structure Type | {model.get('structure_type', '-')} |
| Number of Stories | {model.get('n_stories', '-')} |
| Total Height | {model.get('total_height', 0):.2f} m |
| Number of Nodes | {model.get('n_nodes', '-')} |
| Number of Elements | {model.get('n_elements', '-')} |
| Total Mass | {model.get('total_mass', 0)/1000:.1f} ton |

## Input Motion

| Parameter | Value |
|-----------|-------|
| Record Name | {inp.get('name', '-')} |
| PGA | {inp.get('pga', 0)*100:.0f} gal |
| Duration | {inp.get('duration', 0):.1f} s |
| Time Step | {inp.get('dt', 0)*1000:.1f} ms |

## Key Results

| Metric | Value |
|--------|-------|
| Max Inter-story Drift | {results.get('max_drift', 0)*100:.3f}% |
| Max Displacement | {results.get('max_disp', 0)*100:.1f} cm |
| Max Acceleration | {results.get('max_accel', 0)/9.81:.2f} g |
| Base Shear Coefficient | {results.get('base_shear_coef', 0):.3f} |
| Yielded Elements | {results.get('n_yielded_elements', 0)} |

"""
        
        # Story responses
        if results.get('story_responses'):
            md += "## Story-by-Story Results\n\n"
            md += "| Story | Max Drift (%) | Max Disp (cm) | Ductility |\n"
            md += "|-------|---------------|---------------|----------|\n"
            
            for sr in results['story_responses']:
                md += f"| {sr.get('story', '-')} | {sr.get('max_drift', 0)*100:.3f} | {sr.get('max_disp', 0)*100:.2f} | {sr.get('ductility', 0):.2f} |\n"
            md += "\n"
            
        # Figures
        if self.data['figures']:
            md += "## Figures\n\n"
            for fig in self.data['figures']:
                md += f"![{fig['caption']}]({fig['path']})\n\n*{fig['caption']}*\n\n"
                
        md += f"\n---\n*Generated by {meta.get('generated_by', 'Earthquake Building Simulator')} on {meta.get('date', '')}*\n"
        
        return md
