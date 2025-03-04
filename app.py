import os
import re
import itertools
import subprocess
import logging
import uuid
import io
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import ollama

from flask import Flask, render_template, request, jsonify, session

# ------------------------------------------------------
# Matplotlib Configuration (headless)
# ------------------------------------------------------
plt.switch_backend('Agg')

# ------------------------------------------------------
# Flask Setup
# ------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_default_secret_key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# ------------------------------------------------------
# Global in-memory storage for uploaded files
# ------------------------------------------------------
# Now storing as: {file_id: {"bytes": file_bytes, "ext": file_extension}}
uploaded_files = {}

# ------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# Utility Functions
# ------------------------------------------------------
def allowed_file(filename):
    valid_extensions = {'csv', 'xls', 'xlsx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in valid_extensions

def get_columns_from_df(df):
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        logger.info(f"Columns - Numerical: {numeric_cols}, Categorical: {categorical_cols}")
        return {'numerical': numeric_cols, 'categorical': categorical_cols}
    except Exception as e:
        logger.error(f"Error in get_columns_from_df: {e}")
        return {}

def generate_distribution_charts(df):
    """
    Generate frequency distribution charts with KDE for each numeric column,
    returning a dictionary {column: data_uri} where data_uri is a Base64-encoded PNG.
    """
    charts = {}
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        plt.close('all')
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df,
            x=col,
            kde=True,
            color="#4c72b0",
            stat="density",
            alpha=0.7,
            edgecolor="black"
        )
        plt.title(f"Frequency Distribution with KDE of {col}", fontsize=16, fontweight="bold", color="#333333")
        plt.xlabel(col, fontsize=14, fontweight="medium")
        plt.ylabel("Density", fontsize=14, fontweight="medium")
        plt.grid(which="major", linestyle="--", linewidth=0.6, color="gray", alpha=0.7)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        charts[col] = f"data:image/png;base64,{encoded}"
        logger.info(f"Generated chart for {col}")
        plt.close()
    return charts

def calculate_membership(x, centers, index):
    try:
        if index == 0:
            if x <= centers[index]:
                return 1
            elif x <= centers[index + 1]:
                return (centers[index + 1] - x) / (centers[index + 1] - centers[index])
            else:
                return 0
        elif index == len(centers) - 1:
            if x >= centers[index]:
                return 1
            elif x >= centers[index - 1]:
                return (x - centers[index - 1]) / (centers[index] - centers[index - 1])
            else:
                return 0
        else:
            if x <= centers[index - 1] or x >= centers[index + 1]:
                return 0
            elif x <= centers[index]:
                return (x - centers[index - 1]) / (centers[index] - centers[index - 1])
            else:
                return (centers[index + 1] - x) / (centers[index + 1] - centers[index])
    except Exception as e:
        logger.error(f"Error in calculate_membership: {e}")
        return 0

def apply_fuzzy_logic(file_info, col_partitions, col_ranges, partition_weights):
    """
    Creates fuzzy membership columns for selected columns.
    Returns (DataFrame, list_of_base64_images) where images are generated in memory.
    """
    try:
        ext = file_info.get('ext', '')
        file_bytes = file_info.get('bytes')
        file_io = io.BytesIO(file_bytes)
        if ext.endswith('.csv'):
            df = pd.read_csv(file_io)
        elif ext.endswith('.xls'):
            df = pd.read_excel(file_io, engine='xlrd')
        elif ext.endswith('.xlsx'):
            df = pd.read_excel(file_io, engine='openpyxl')
        else:
            raise ValueError("Unsupported file format")
        # Keep only the columns configured for partitioning
        df = df[list(col_partitions.keys())]
        fuzzy_df = pd.DataFrame(index=df.index)
        membership_plots = []
        for col, num_parts in col_partitions.items():
            user_min, user_max = col_ranges[col]
            actual_min = float(user_min) if user_min is not None else float(df[col].min())
            actual_max = float(user_max) if user_max is not None else float(df[col].max())
            if num_parts > 1:
                interval = (actual_max - actual_min) / (num_parts - 1)
                centers = [actual_min + i * interval for i in range(num_parts)]
            else:
                centers = [actual_min]
            # Compute fuzzy membership values for each partition
            for i in range(num_parts):
                part_col = f"{col}_P{i+1}"
                fuzzy_df[part_col] = df[col].apply(lambda val: calculate_membership(val, centers, i))
            # Generate membership plot for this column
            x_vals = np.linspace(actual_min, actual_max, 500)
            plt.figure(figsize=(8, 6))
            for i in range(num_parts):
                y_vals = [calculate_membership(x, centers, i) for x in x_vals]
                plt.plot(x_vals, y_vals, label=f"{col}_P{i+1}")
            plt.title(f"Fuzzy Membership for {col}")
            plt.xlabel("Value Range")
            plt.ylabel("Membership Degree")
            plt.legend()
            plt.grid(True)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
            membership_plots.append(f"data:image/png;base64,{encoded}")
            logger.info(f"Generated fuzzy membership plot for {col}")
        return fuzzy_df, membership_plots
    except Exception as e:
        logger.error(f"Error in apply_fuzzy_logic: {e}")
        raise

def calculate_significance(combo, df, weights, min_support=0.15):
    """
    For a combo of columns (e.g., col_P1, col_P2), if membership*weight < min_support => 0 for that row.
    Else multiply in. Final significance = average of row products.
    """
    try:
        total = len(df)
        sum_sig = 0.0

        for _, row in df.iterrows():
            product = 1.0
            for col_partition in combo:
                membership_val = float(row[col_partition])
                base_col = col_partition.split("_P")[0]
                weight = float(weights.get(base_col, {}).get(col_partition, 1.0))
                if membership_val * weight >= min_support:
                    product *= (membership_val * weight)
                else:
                    product = 0
                    break
            sum_sig += product

        significance = (sum_sig / total) if total else 0.0
        return significance
    except Exception as e:
        logger.error(f"Error in calculate_significance: {e}")
        return 0.0

def determine_farm_rules(df, weights, min_support=0.15):
    """
    Multi-phase itemset generation until no combos remain.
    Only combos from distinct base columns. Return (frequent_itemsets, step_details).
    """
    try:
        cols = df.columns
        kept = set(cols)
        step_details = []
        all_itemsets = []

        for phase in range(1, len(cols) + 1):
            phase_info = {"phase": phase, "kept": [], "pruned": []}
            combos = [
                c for c in itertools.combinations(kept, phase)
                if len({cc.split("_P")[0] for cc in c}) == len(c)
            ]

            newly_kept = []
            for combo in combos:
                sig = calculate_significance(combo, df, weights, min_support=min_support)
                if sig >= min_support:
                    newly_kept.append(combo)
                    phase_info["kept"].append({"combination": combo, "significance": sig})
                else:
                    phase_info["pruned"].append({"combination": combo, "significance": sig})

            # Update
            kept = {item for combo in newly_kept for item in combo}
            all_itemsets.extend(newly_kept)
            step_details.append(phase_info)

            logger.debug(f"Phase {phase}: Kept {len(newly_kept)} combos, Pruned {len(phase_info['pruned'])} combos.")

            if not newly_kept:
                break

        return all_itemsets, step_details
    except Exception as e:
        logger.error(f"Error in determine_farm_rules: {e}")
        return [], []

def generate_rules(itemsets, df, weights, min_certainty=0.60):
    """
    From itemsets => FARM rules if certainty >= min_certainty.
    """
    try:
        def calc_certainty(antecedent, consequent):
            union_c = tuple(set(antecedent) | {consequent})
            sig_union = calculate_significance(union_c, df, weights)
            sig_antecedent = calculate_significance(antecedent, df, weights)
            return (sig_union / sig_antecedent) if sig_antecedent else 0

        rules = []
        for itemset in itemsets:
            if len(itemset) < 2:  # need at least 2 items
                continue
            for c in itemset:
                a = tuple(set(itemset) - {c})
                cf = calc_certainty(a, c)
                if cf >= min_certainty:
                    rules.append({
                        "antecedent": a,
                        "consequent": c,
                        "certainty": cf
                    })
        logger.debug(f"Generated {len(rules)} FARM rules.")
        return rules
    except Exception as e:
        logger.error(f"Error in generate_rules: {e}")
        return []

def convert_farm_rules_html_to_english(farm_rules_html):
    """
    Converts FARM rules from an HTML table to English sentences.
    """
    try:
        soup = BeautifulSoup(farm_rules_html, 'html.parser')
        rules = []
        table = soup.find('table', {'class': 'farm-rules-table'})
        if not table:
            raise ValueError("No table found with class 'farm-rules-table'.")
        headers = [th.get_text(strip=True).lower() for th in table.find('thead').find_all('th')]
        required_headers = {'antecedent', 'consequent', 'certainty'}
        if not required_headers.issubset(set(headers)):
            raise ValueError(f"Table headers missing required columns: {required_headers}")
        header_indices = {header: index for index, header in enumerate(headers)}
        for row in table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) < len(required_headers):
                continue
            antecedent = cells[header_indices['antecedent']].get_text(strip=True)
            consequent = cells[header_indices['consequent']].get_text(strip=True)
            certainty = cells[header_indices['certainty']].get_text(strip=True)
            try:
                certainty_value = float(certainty)
                certainty_percentage = f"{certainty_value * 100:.1f}%"
            except ValueError:
                certainty_percentage = certainty
            sentence = f"If {antecedent}, then {consequent} with a certainty of {certainty_percentage}."
            rules.append(sentence)
        logger.info("Converted HTML FARM rules to English sentences.")
        return rules
    except Exception as e:
        logger.error(f"Error in convert_farm_rules_html_to_english: {e}")
        return []

def convert_farm_rules_html_to_structured_data(farm_rules_html):
    """
    Converts FARM rules from HTML table format to a structured data format.

    Args:
        farm_rules_html (str): HTML string containing the FARM rules table.

    Returns:
        List[dict]: A list of dictionaries, each representing a FARM rule with
                    keys: 'antecedent', 'consequent', and 'certainty'.
    """
    try:
        soup = BeautifulSoup(farm_rules_html, 'html.parser')
        structured_rules = []

        table = soup.find('table', {'class': 'farm-rules-table'})
        if not table:
            raise ValueError("No table found with class 'farm-rules-table'.")

        headers = [th.get_text(strip=True).lower() for th in table.find('thead').find_all('th')]
        required_headers = {'antecedent', 'consequent', 'certainty'}
        if not required_headers.issubset(set(headers)):
            raise ValueError(f"Table headers missing required columns: {required_headers}")

        header_indices = {header: index for index, header in enumerate(headers)}

        for row in table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) < len(required_headers):
                continue

            antecedent = cells[header_indices['antecedent']].get_text(strip=True)
            consequent = cells[header_indices['consequent']].get_text(strip=True)
            certainty = cells[header_indices['certainty']].get_text(strip=True)

            antecedent_list = [item.strip() for item in antecedent.split(',') if item.strip()]
            try:
                certainty_value = float(certainty)
            except ValueError:
                logger.warning(f"Invalid certainty value: {certainty}. Skipping rule.")
                continue

            structured_rules.append({
                'antecedent': antecedent_list,
                'consequent': consequent,
                'certainty': certainty_value
            })

        logger.debug(f"Converted HTML FARM rules to structured data. Total rules: {len(structured_rules)}")
        return structured_rules
    except Exception as e:
        logger.error(f"Error in convert_farm_rules_html_to_structured_data: {e}")
        return []

# ------------------------------------------------------
# Flask Routes
# ------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('base.html')

@app.route('/ajax_upload', methods=['POST'])
def ajax_upload():
    try:
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            logger.info("Invalid file format uploaded.")
            return jsonify({"success": False, "error": "Invalid file format. Please upload CSV or Excel."}), 400

        file_data = file.read()
        unique_id = uuid.uuid4().hex
        # Store both file bytes and extension together
        uploaded_files[unique_id] = {"bytes": file_data, "ext": file.filename.lower()}
        session['uploaded_file_id'] = unique_id
        session['uploaded_file_ext'] = file.filename.lower()  # optional for backward compatibility

        # Read file into DataFrame based on its extension
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_data), engine='python')
        elif file.filename.lower().endswith('.xls'):
            df = pd.read_excel(io.BytesIO(file_data), engine='xlrd')
        elif file.filename.lower().endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(file_data), engine='openpyxl')
        else:
            raise ValueError("Unsupported file format")
        columns = get_columns_from_df(df)
        charts = generate_distribution_charts(df)

        auto_ranges = {}
        for col in columns.get('numerical', []):
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            auto_ranges[col] = {"min": round(col_min, 3), "max": round(col_max, 3)}

        logger.info("File processing completed successfully.")
        return jsonify({
            "success": True,
            "uploaded_file_id": unique_id,
            "columns": columns,
            "auto_ranges": auto_ranges,
            "chart_paths": charts,
        })
    except Exception as e:
        logger.error(f"Error in ajax_upload: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/ajax_process', methods=['POST'])
def ajax_process():
    try:
        file_id = session.get('uploaded_file_id')
        if not file_id or file_id not in uploaded_files:
            logger.info("No valid uploaded file available in ajax_process route.")
            return jsonify({"success": False, "error": "No valid file available."}), 400

        # Retrieve the file info (bytes and extension)
        file_info = uploaded_files[file_id]

        selected_cols = request.form.getlist('columns')
        if not selected_cols:
            logger.info("No columns selected for processing.")
            return jsonify({"success": False, "error": "No columns selected for processing."}), 400

        min_support = session.get('min_support', 0.15)
        min_certainty = session.get('min_certainty', 0.6)
        logger.info(f"Using thresholds => min_support: {min_support}, min_certainty: {min_certainty}")

        partitions_map, ranges_map, weights_map = {}, {}, {}

        for col in selected_cols:
            part_count = int(request.form.get(f"partitions_{col}", 3))
            range_type = request.form.get(f"range_type_{col}", 'auto')
            user_min = request.form.get(f"min_{col}")
            user_max = request.form.get(f"max_{col}")
            if range_type == 'manual' and user_min and user_max:
                col_min = float(user_min)
                col_max = float(user_max)
            else:
                col_min = col_max = None
            partitions_map[col] = part_count
            ranges_map[col] = (col_min, col_max)
            weights_map[col] = {}
            for i in range(1, part_count + 1):
                w_key = f"weight_{col}_P{i}"
                w_val = request.form.get(w_key)
                try:
                    weights_map[col][f"{col}_P{i}"] = float(w_val) if w_val else 1.0
                except ValueError:
                    weights_map[col][f"{col}_P{i}"] = 1.0

        fuzzy_df, membership_plots = apply_fuzzy_logic(file_info, partitions_map, ranges_map, weights_map)

        itemsets, step_details = determine_farm_rules(fuzzy_df, weights_map, min_support=min_support)
        rules = generate_rules(itemsets, fuzzy_df, weights_map, min_certainty=min_certainty)

        rules_html = "<table class='farm-rules-table'><thead><tr><th>Antecedent</th><th>Consequent</th><th>Certainty</th></tr></thead><tbody>"
        for rule in rules:
            antecedent = ', '.join(rule['antecedent'])
            consequent = rule['consequent']
            certainty = f"{rule['certainty']:.4f}"
            rules_html += f"<tr><td>{antecedent}</td><td>{consequent}</td><td>{certainty}</td></tr>"
        rules_html += "</tbody></table>"

        # Store the FARM rules HTML in session for later use
        session['farm_rules_html'] = rules_html

        results_html = fuzzy_df.to_html(classes='table table-bordered', index=False)
        steps_html = ""
        for phase in step_details:
            p_num = phase["phase"]
            steps_html += f"<div><h4>Phase {p_num}: {p_num}-Itemsets</h4>"
            steps_html += "<table class='combinations-table'><thead><tr><th>Combination</th><th>Significance</th><th>Status</th></tr></thead><tbody>"
            for k in phase["kept"]:
                combination = ', '.join(k['combination'])
                significance = f"{k['significance']:.4f}"
                steps_html += f"<tr><td>{combination}</td><td>{significance}</td><td style='color:green;'>Kept</td></tr>"
            for p in phase["pruned"]:
                combination = ', '.join(p['combination'])
                significance = f"{p['significance']:.4f}"
                steps_html += f"<tr><td>{combination}</td><td>{significance}</td><td style='color:red;'>Pruned</td></tr>"
            steps_html += "</tbody></table></div>"

        session['selected_columns'] = selected_cols
        session['partitions_map'] = partitions_map
        session['ranges_map'] = ranges_map
        session['weights_map'] = weights_map

        logger.info("AJAX processing completed successfully.")
        return jsonify({
            "success": True,
            "results_html": results_html,
            "rules_html": rules_html,
            "steps_html": steps_html,
            "plots": membership_plots
        })
    except Exception as e:
        logger.error(f"Error in ajax_process: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/explore_farm_rules', methods=['POST'])
def explore_farm_rules():
    """
    Explore FARM rules by varying support and certainty thresholds.
    Generates a heatmap of the number of rules generated and returns it as a Base64-encoded image.
    """
    try:
        logger.debug("Starting explore_farm_rules...")

        step_support = float(request.form.get('step_support', 0.1))
        step_certainty = float(request.form.get('step_certainty', 0.1))
        logger.debug(f"Received step sizes -> step_support: {step_support}, step_certainty: {step_certainty}")

        # Retrieve the file info using the stored uploaded_file_id
        file_id = session.get('uploaded_file_id')
        if not file_id or file_id not in uploaded_files:
            logger.error("No valid file available for exploring FARM rules.")
            return jsonify({"success": False, "error": "No valid file available."}), 400
        file_info = uploaded_files[file_id]

        selected_cols = session.get('selected_columns', [])
        partitions_map = session.get('partitions_map', {})
        ranges_map = session.get('ranges_map', {})
        weights_map = session.get('weights_map', {})
        logger.debug(f"Selected columns: {selected_cols}")
        logger.debug(f"Partitions map: {partitions_map}")
        logger.debug(f"Ranges map: {ranges_map}")
        logger.debug(f"Weights map: {weights_map}")

        # Apply fuzzy logic using the file info
        fuzzy_df, _ = apply_fuzzy_logic(file_info, partitions_map, ranges_map, weights_map)

        supports = np.arange(session.get('min_support', 0.2), 1.01, step_support)
        certainties = np.arange(session.get('min_certainty', 0.5), 1.01, step_certainty)
        logger.debug(f"Supports: {supports}")
        logger.debug(f"Certainties: {certainties}")

        heatmap_data = np.zeros((len(supports), len(certainties)))
        rule_details = []

        for i, support in enumerate(supports):
            for j, certainty in enumerate(certainties):
                itemsets, _ = determine_farm_rules(fuzzy_df, weights_map, min_support=support)
                rules = generate_rules(itemsets, fuzzy_df, weights_map, min_certainty=certainty)
                heatmap_data[i, j] = len(rules)
                rule_details.append({
                    "support": round(support, 2),
                    "certainty": round(certainty, 2),
                    "rule_count": len(rules),
                    "rules": rules,
                })

        # Generate the heatmap and encode it as Base64 without saving to disk.
        plt.figure(figsize=(10, 8))

        # 1) Display the heatmap with the normal "Blues" colormap
        bg_cmap = plt.cm.cool
        im = plt.imshow(heatmap_data, cmap=bg_cmap, interpolation="nearest", origin="lower")
        plt.colorbar(im, label="Number of FARM Rules")
        plt.xticks(range(len(certainties)), [f"{c:.2f}" for c in certainties], rotation=45)
        plt.yticks(range(len(supports)), [f"{s:.2f}" for s in supports])
        plt.xlabel("Certainty Threshold")
        plt.ylabel("Support Threshold")
        plt.title("FARM Rules Heatmap")

        # 2) Create a normalization for the data (maps data values to [0..1])
        import matplotlib as mpl
        norm = mpl.colors.Normalize(vmin=heatmap_data.min(), vmax=heatmap_data.max())

        # 3) Loop over each cell, compute background color luminance, and pick text color
        for i in range(len(supports)):
            for j in range(len(certainties)):
                val = int(heatmap_data[i, j])
                nv = norm(val)
                # Get the background color from the same colormap
                r, g, b, a = bg_cmap(nv)

                # Calculate luminance (brightness)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b

                # If luminance is below threshold, choose white text; otherwise black
                # (Adjust threshold to "give more weight to white," e.g., 0.6)
                if luminance < 0.6:
                    text_color = "white"
                else:
                    text_color = "black"

                plt.text(j, i, val,
                         ha="center", va="center",
                         color=text_color, fontsize=20)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        encoded_heatmap = base64.b64encode(buf.getvalue()).decode('utf-8')
        heatmap_data_uri = f"data:image/png;base64,{encoded_heatmap}"

        logger.debug("Heatmap generated and encoded as Base64.")

        response_data = {
            "success": True,
            "summary": f"Explored {len(supports) * len(certainties)} combinations.",
            "details": "<h3>Detailed FARM Rules</h3>",
            "heatmap": heatmap_data_uri,
        }
        logger.debug(f"Response data: {response_data}")

        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in explore_farm_rules: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get_models', methods=['GET'])
def get_models():
    """
    Fetch available AI models via ollama.list() and return model names as JSON.
    """
    try:
        logger.debug("Entered get_models route")
        model_data = str(ollama.list())
        pattern = r"model='(.*?)'"
        models = re.findall(pattern, model_data)
        models = [name for name in models if name.strip()]
        logger.debug(f"Available models: {models}")
        return jsonify({"success": True, "models": models})
    except Exception as e:
        logger.error(f"Error in get_models: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/set_model', methods=['POST'])
def set_model():
    try:
        selected_model = request.form.get('model')
        if not selected_model:
            logger.warning("No model selected in set_model route.")
            return jsonify({"success": False, "error": "No model selected."}), 400
        session['selected_model'] = selected_model
        logger.info(f"Selected Ollama model set to: {selected_model}")
        return jsonify({"success": True, "message": f"Model '{selected_model}' set successfully."})
    except Exception as e:
        logger.error(f"Error in set_model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/set_thresholds', methods=['POST'])
def set_thresholds():
    try:
        min_support = request.form.get('min_support', type=float)
        min_certainty = request.form.get('min_certainty', type=float)
        if min_support is None or min_certainty is None:
            logger.warning("Thresholds not properly set in set_thresholds route.")
            return jsonify({"success": False, "error": "Invalid thresholds provided."}), 400
        session['min_support'] = min_support
        session['min_certainty'] = min_certainty
        logger.info(f"Updated thresholds => min_support: {min_support}, min_certainty: {min_certainty}")
        return jsonify({
            "success": True,
            "min_support": min_support,
            "min_certainty": min_certainty
        })
    except Exception as e:
        logger.error(f"Error in set_thresholds: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    """
    Generate a summary of FARM rules using the selected Ollama AI model.
    Rules are categorized by antecedent count, sorted by certainty,
    and limited to the top 10 rules per category.
    """
    try:
        # Retrieve FARM rules HTML from session
        farm_rules_html = session.get('farm_rules_html')
        if not farm_rules_html:
            logger.warning("No FARM rules available for summary generation.")
            return jsonify({"success": False, "error": "No FARM rules available. Please process data first."}), 400

        # Convert HTML FARM rules to structured data
        rules = convert_farm_rules_html_to_structured_data(farm_rules_html)
        if not rules:
            logger.warning("No valid FARM rules found to summarize.")
            return jsonify({"success": False, "error": "No valid FARM rules found to summarize."}), 400

        # Categorize rules by antecedent count and sort by certainty
        categorized_rules = {}
        for rule in rules:
            num_antecedents = len(rule['antecedent'])
            certainty = rule['certainty']
            categorized_rules.setdefault(num_antecedents, []).append((rule, certainty))

        # Sort each category by certainty in descending order and limit to top 10
        for key in categorized_rules:
            categorized_rules[key] = sorted(categorized_rules[key], key=lambda x: x[1], reverse=True)[:10]

        prompts = []
        for num_antecedents in sorted(categorized_rules.keys(), reverse=True):
            category_header = f"Rules with {num_antecedents} antecedent(s):\n"
            rules_text = "\n\n".join([
                f"{idx + 1}. IF {', '.join(rule['antecedent'])} THEN {rule['consequent']} [Certainty: {certainty:.4f}]"
                for idx, (rule, certainty) in enumerate(categorized_rules[num_antecedents])
            ])
            prompts.append(f"{category_header}\n{rules_text}\n")

        full_prompt = "\n\n".join(prompts)

        default_prompt_template = (
            "We are conducting Fuzzy Associative Rule Mining (FARM) using the Apriori algorithm. "
            "The rules have been categorized based on the number of antecedents and arranged "
            "in descending order by the number of antecedents, then further sorted by certainty. "
            "Here are the top 10 rules for each category. Kindly analyze these rules and provide a concise summary. "
            "Ensure the summary captures the essential relationships, key patterns, and insights reflected by the rules. "
            "Write in a systematic fashion with clear line breaks for readability. EXPLAIN EACH AND EVERY RULE::::\n\n{rules}"
        )

        full_prompt = default_prompt_template.format(rules=full_prompt)

        logger.debug("Generated categorized and sorted FARM rules for summarization with top 10 rules per category.")

        selected_model = session.get('selected_model')
        if not selected_model:
            logger.warning("No Ollama model selected for summary generation.")
            return jsonify({"success": False, "error": "No Ollama model selected. Please set a model first."}), 400

        modified_prompt = session.get('modified_prompt')
        if modified_prompt:
            if "{rules}" not in modified_prompt:
                prompt = f"{modified_prompt}\n\n{full_prompt}"
                logger.debug("Using modified prompt without placeholder.")
            else:
                prompt = modified_prompt.format(rules=full_prompt)
                logger.debug("Using modified prompt from session.")
        else:
            prompt = full_prompt
            logger.debug("Using improved default prompt for summary generation.")

        logger.debug(f"Sending prompt to Ollama model '{selected_model}': {prompt[:100]}...")
        response = ollama.chat(model=selected_model, messages=[{'role': 'user', 'content': prompt}])
        summary = response['message']['content']
        if not summary:
            logger.error("Received empty summary from Ollama model.")
            return jsonify({"success": False, "error": "Received empty summary from Ollama model."}), 500

        logger.debug("Summary generated successfully.")
        return jsonify({"success": True, "summary": summary})
    except Exception as e:
        logger.error(f"Error in generate_summary: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/set_modified_prompt', methods=['POST'])
def set_modified_prompt():
    modified_prompt = request.form.get('modified_prompt', '').strip()
    if not modified_prompt:
        logger.warning("Empty prompt received in set_modified_prompt route.")
        return jsonify({"success": False, "error": "Prompt cannot be empty."}), 400
    if "{rules}" not in modified_prompt:
        logger.warning("Invalid prompt received. Missing '{rules}' placeholder.")
        return jsonify({"success": False, "error": "Prompt must include the '{rules}' placeholder."}), 400
    session['modified_prompt'] = modified_prompt
    logger.info("Modified prompt set successfully.")
    return jsonify({"success": True, "message": "Prompt modified successfully."})

@app.route('/get_current_prompt', methods=['GET'])
def get_current_prompt():
    try:
        if 'farm_rules_html' not in session:
            logger.warning("No FARM rules available to generate prompt.")
            return jsonify({"success": False, "error": "No FARM rules available. Please process data first."}), 400
        selected_model = session.get('selected_model')
        if not selected_model:
            logger.warning("No Ollama model selected to generate prompt.")
            return jsonify({"success": False, "error": "No Ollama model selected. Please set a model first."}), 400
        modified_prompt = session.get('modified_prompt')
        prompt = modified_prompt if modified_prompt else (
            "We are conducting Fuzzy Associative Rule Mining (FARM) using the Apriori algorithm. "
            "Here are the top 10 rules for each category. Kindly analyze these rules and provide a concise summary. "
            "Ensure the summary captures the essential relationships and insights. EXPLAIN EACH AND EVERY RULE::::\n\n{rules}"
        )
        logger.info("Returning current prompt.")
        return jsonify({"success": True, "prompt": prompt})
    except Exception as e:
        logger.error(f"Error in get_current_prompt: {e}")
        return jsonify({"success": False, "error": "An error occurred while retrieving the current prompt."}), 500

@app.route('/reset_modified_prompt', methods=['POST'])
def reset_modified_prompt():
    try:
        session.pop('modified_prompt', None)
        logger.info("Modified prompt reset successfully.")
        return jsonify({"success": True, "message": "Prompt reset to default."})
    except Exception as e:
        logger.error(f"Error in reset_modified_prompt: {e}")
        return jsonify({"success": False, "error": "An error occurred while resetting the modified prompt."}), 500

def is_ollama_running():
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Ollama check failed: {e}")
        return False

@app.route('/check_ai_readiness', methods=['GET'])
def check_ai_readiness():
    if not is_ollama_running():
        return jsonify({
            "ollama_ready": False,
            "models": [],
            "error": "Ollama is not running or not found in PATH."
        })
    try:
        model_data = str(ollama.list())
        pattern = r"model='(.*?)'"
        models = re.findall(pattern, model_data)
        models = [name.strip() for name in models if name.strip()]
        logger.info(f"Installed Ollama AI Models: {models}")
        return jsonify({
            "ollama_ready": True,
            "models": models
        })
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {e}")
        return jsonify({
            "ollama_ready": True,
            "models": [],
            "error": f"Error fetching Ollama models: {e}"
        })

# ------------------------------------------------------
# Entry Point
# ------------------------------------------------------
if __name__ == '__main__':
    app.run()
