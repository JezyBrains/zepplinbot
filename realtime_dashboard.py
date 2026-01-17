#!/usr/bin/env python3
"""
Zeppelin Pro Dashboard - Premium Trading Interface
Stunning glassmorphism design with advanced visualizations
"""

import numpy as np
import pandas as pd
import datetime
import os
import sys
import json
import requests
import time
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_socketio import SocketIO, emit

# Import V2 modules at top level to avoid callback lag
try:
    from feature_engineering import feature_engine
    from regime_detector import betting_window_detector
except ImportError:
    # Handle the case where src is not in path yet
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from feature_engineering import feature_engine
    from regime_detector import betting_window_detector



sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import dash_bootstrap_components as dbc
    import plotly.graph_objs as go
except ImportError:
    print("Install: pip install dash dash-bootstrap-components plotly")
    exit(1)

# Import advanced predictor
if os.getenv('DISABLE_ADVANCED_PREDICTOR') == 'true':
    ADVANCED_PREDICTOR = False
    print("⚠️ Advanced Predictor DISABLED via Environment Variable")
else:
    try:
        from src.advanced_predictor import get_advanced_signal, predictor
        ADVANCED_PREDICTOR = True
    except:
        ADVANCED_PREDICTOR = False

# Import translations
try:
    from src.translations import TRANSLATIONS
except ImportError:
    TRANSLATIONS = {'en': {}, 'sw': {}}


# ============ DATA ============
# Ensure data directory exists BEFORE any file operations
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

crash_data = []
bankroll = 100000.0
DATA_FILE = 'data/crash_data.csv'
ROUND_DATA_FILE = 'data/round_timing.csv'
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '') # Set in .env


def load_data():
    global crash_data
    try:
        if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
            try:
                df = pd.read_csv(DATA_FILE)
                if 'value' in df.columns:
                    crash_data = df['value'].tolist()
            except pd.errors.EmptyDataError:
                pass
        
        # Ensure round_timing.csv columns are handled gracefully
        if os.path.exists(ROUND_DATA_FILE) and os.path.getsize(ROUND_DATA_FILE) > 10:
            try:
                df_timing = pd.read_csv(ROUND_DATA_FILE, on_bad_lines='skip', low_memory=False)
                for col in ['total_won', 'velocity_metrics', 'seeds']:
                    if col not in df_timing.columns:
                        df_timing[col] = '{}' if 'metrics' in col or 'seeds' in col else 0
            except:
                pass
    except Exception as e:
        print(f"Error loading data: {e}")

def save_data():
    global crash_data
    try:
        os.makedirs('data', exist_ok=True)
        # Create timestamps if needed (simple fallback)
        timestamps = [datetime.now().isoformat()] * len(crash_data)
        pd.DataFrame({'timestamp': timestamps, 'value': crash_data}).to_csv(DATA_FILE, index=False)
    except Exception as e:
        print(f"Error saving data: {e}")

def send_alert(message, title="🚨 Zeppelin Alert"):
    """Send alert to Discord if webhook is configured"""
    if not DISCORD_WEBHOOK_URL:
        return
    
    payload = {
        "embeds": [{
            "title": title,
            "description": message,
            "color": 0x00d97e if "WIN" in title else 0xe63757,
            "timestamp": datetime.now().isoformat()
        }]
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"Failed to send alert: {e}")


def get_signal():
    """
    Unified signal generation using the AdvancedPredictor ensemble.
    """
    global bankroll
    if not ADVANCED_PREDICTOR:
        return {'signal': 'WAIT', 'reason': 'Predictor not loaded', 'color': '#ffc107', 'analysis': {}}
        
    predictor.bankroll = bankroll
    # Update predictor history from the global list
    predictor.history = crash_data[:]
    
    # Get the full ensemble signal
    sig = predictor.get_combined_signal()
    
    # Add color mapping for UI
    signal = sig.get('signal', 'WAIT')
    sig['color'] = '#10b981' if signal == 'BET' else '#e63757' if signal == 'SKIP' else '#ffc107'
    
    return sig


def get_temporal_insights(timeframe='all'):
    """
    Analyze round_timing.csv to compute temporal insights for the dashboard.
    Supports timeframe filtering and weighted analysis.
    """
    insights = {
        'hourly_avg': {h: 0.0 for h in range(24)},
        'hourly_count': {h: 0 for h in range(24)},
        'instant_crash_rate': {h: 0.0 for h in range(24)},
        'best_hour': 0,
        'best_hour_avg': 0.0,
        'worst_hour': 0,
        'worst_hour_avg': 0.0,
        'velocity_anomalies': [],
        'total_rounds': 0,
        'high_stake_warning': False,
        'high_stake_diff': 0.0
    }
    
    try:
        if not os.path.exists(ROUND_DATA_FILE):
            return insights
            
        df = pd.read_csv(ROUND_DATA_FILE, on_bad_lines='skip', low_memory=False)
        if len(df) < 5:
            return insights
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # Apply Timeframe Filtering (Ensure timeframe is valid)
        if not timeframe:
            timeframe = 'all'
            
        now = datetime.now()
        if timeframe == '24h':
            df = df[df['timestamp'] > (now - timedelta(days=1))]
        elif timeframe == '7d':
            df = df[df['timestamp'] > (now - timedelta(days=7))]
        
        if len(df) == 0:
            return insights

        insights['total_rounds'] = len(df)
        
        if timeframe == 'weighted':
            # Weighted Analysis (Inverse time decay)
            max_ts = df['timestamp'].max()
            df['weight'] = df['timestamp'].apply(lambda tx: 1.0 / (1.0 + (max_ts - tx).total_seconds() / 86400)) # Decays over days
            
            for hour in range(24):
                hr_data = df[df['hour'] == hour]
                if len(hr_data) > 0:
                    insights['hourly_avg'][hour] = round((hr_data['crash_value'] * hr_data['weight']).sum() / hr_data['weight'].sum(), 2)
                    insights['hourly_count'][hour] = len(hr_data)
                    # Weighted instant crash rate
                    instant_mask = hr_data['crash_value'] <= 1.1
                    insights['instant_crash_rate'][hour] = round((instant_mask * hr_data['weight']).sum() / hr_data['weight'].sum() * 100, 1)
        else:
            # Standard Averages
            hourly_stats = df.groupby('hour')['crash_value'].agg(['mean', 'count'])
            for hour in hourly_stats.index:
                insights['hourly_avg'][hour] = round(hourly_stats.loc[hour, 'mean'], 2)
                insights['hourly_count'][hour] = int(hourly_stats.loc[hour, 'count'])
            
            # Instant crash rate by hour
            for hour in df['hour'].unique():
                hour_data = df[df['hour'] == hour]
                instant_rate = (hour_data['crash_value'] <= 1.1).sum() / len(hour_data)
                insights['instant_crash_rate'][int(hour)] = round(instant_rate * 100, 1)
        
        # Best and worst hours (using computed averages)
        avg_list = [(h, insights['hourly_avg'][h]) for h in range(24) if insights['hourly_count'][h] > 0]
        if avg_list:
            best = max(avg_list, key=lambda x: x[1])
            worst = min(avg_list, key=lambda x: x[1])
            insights['best_hour'] = int(best[0])
            insights['best_hour_avg'] = best[1]
            insights['worst_hour'] = int(worst[0])
            insights['worst_hour_avg'] = worst[1]
        
        # High stake analysis (only on filtered data)
        if 'stake' in df.columns and len(df) > 20:
            stake_clean = df[df['stake'] > 0]['stake']
            if len(stake_clean) > 20:
                high_threshold = stake_clean.quantile(0.9)
                high_stake = df[df['stake'] >= high_threshold]
                normal_stake = df[(df['stake'] > 0) & (df['stake'] < high_threshold)]
                if len(high_stake) > 5 and len(normal_stake) > 5:
                    diff = normal_stake['crash_value'].mean() - high_stake['crash_value'].mean()
                    if diff > 1.0:
                        insights['high_stake_warning'] = True
                        insights['high_stake_diff'] = round(diff, 2)
        
        # Velocity anomalies (always from most recent data regardless of timeframe)
        recent = df.tail(100)
        anomalies = []
        for _, row in recent.iterrows():
            try:
                vm_str = row.get('velocity_metrics', '[]')
                if pd.isna(vm_str) or vm_str in ['{}', '[]', '']:
                    continue
                vm = json.loads(str(vm_str).replace("''", '"').replace('""', '"'))
                if not isinstance(vm, list) or len(vm) < 5:
                    continue
                deltas = [e['delta_ms'] for e in vm if 'delta_ms' in e and e['delta_ms'] > 0]
                if len(deltas) >= 5:
                    avg_delta = np.mean(deltas[:-3])
                    last_avg = np.mean(deltas[-3:])
                    if avg_delta > 0 and last_avg < avg_delta * 0.5:
                        anomalies.append({
                            'round_id': row.get('round_id', 'Unknown'),
                            'crash_value': row.get('crash_value', 0),
                            'speedup': round(last_avg / avg_delta * 100, 1)
                        })
            except:
                continue
        insights['velocity_anomalies'] = anomalies[-5:]
        
    except Exception as e:
        print(f"Temporal insights error: {e}")
    
    return insights


# REAL-TIME MODE: Don't load historical data - only use incoming live data
# apply to the server (flask app) - trust 2 layers (Traefik + Dokploy)
server.wsgi_app = ProxyFix(server.wsgi_app, x_for=2, x_proto=1, x_host=1, x_prefix=1)

@server.before_request
def debug_ip():
    if request.path.startswith('/api/'):
        print(f"🔍 IP DEBUG: remote_addr={request.remote_addr} | Headers={dict(request.headers)}")

print("🔴 REAL-TIME MODE: Waiting for live data from extension...")

# ============ APP ============
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=IBM+Plex+Mono:wght@400;700&display=swap',
    dbc.themes.BOOTSTRAP
]
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, 'https://use.fontawesome.com/releases/v5.15.4/css/all.css'], suppress_callback_exceptions=True)
server = app.server
socketio = SocketIO(server, cors_allowed_origins="*", async_mode='gevent')

@socketio.on('connect')
def handle_connect():
    print(f"📡 Client connected: {datetime.datetime.now()}")
    emit('status', {'data': 'Connected to Zeppelin WebSocket'})

@socketio.on('crash_report')
def handle_crash_report(data):
    val = data.get('value')
    if val:
        print(f"🎯 Real-time Crash Received via Socket: {val}x")
        if val not in crash_data:
            crash_data.append(float(val))
            save_data()
            socketio.emit('new_crash_ack', {'status': 'saved', 'value': val})

server.wsgi_app = ProxyFix(server.wsgi_app, x_for=2)

@server.after_request
def cors(r):
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return r


# Track data source
last_ip = "WAITING..."

@server.route('/api/crash', methods=['POST', 'OPTIONS'])
def api_crash():
    global last_ip
    if request.method == 'OPTIONS': return '', 200
    try:
        # ProxyFix handles X-Forwarded-For automatically
        last_ip = request.remote_addr
        v = float(request.get_json().get('value', 0))
        if v >= 1.0 and v not in crash_data[-5:]:
            crash_data.append(v)
            save_data()
            return jsonify({'success': True, 'total': len(crash_data)})
        return jsonify({'success': True, 'duplicate': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@server.route('/api/signal')
def api_signal():
    # Real-time mode: use current crash_data, don't reload from CSV
    return jsonify(get_signal())

@server.route('/api/round-data', methods=['POST', 'OPTIONS'])
def api_round_data():
    if request.method == 'OPTIONS': return '', 200
    try:
        data = request.get_json()
        os.makedirs('data', exist_ok=True)
        
        # Prepare row for CSV with EXPLICIT order to match header
        row = {
            'timestamp': datetime.now().isoformat(),
            'round_id': data.get('roundId'),
            'duration_ms': data.get('durationMs'),
            'crash_value': data.get('crashValue'),
            'bettors': data.get('bettorsAtStart'),
            'stake': data.get('stakeAtStart'),
            'total_won': data.get('totalWinTZS', 0),
            'cashout_ratio': data.get('cashoutRatio'),
            'cashout_events': json.dumps(data.get('cashoutEvents', [])),
            'velocity_metrics': json.dumps(data.get('velocityMetrics', [])),
            'seeds': json.dumps(data.get('seeds', {}))
        }

        
        # Append to CSV
        file_exists = os.path.exists(ROUND_DATA_FILE)
        df = pd.DataFrame([row])
        df.to_csv(ROUND_DATA_FILE, mode='a', index=False, header=not file_exists)

        # 🚨 ALERT TRIGGERS (Phase 11)
        try:
            stake = row.get('stake', 0)
            won = row.get('total_won', 0)
            payout_ratio = (won / stake * 100) if stake > 0 else 0
            
            if payout_ratio > 150 and stake > 1000:
                send_alert(
                    f"🎰 **WHALE WIN DETECTED!**\n"
                    f"Round ID: `{row['round_id']}`\n"
                    f"Stake: `{stake:,.0f} TZS`\n"
                    f"Payout: `{won:,.0f} TZS` ({payout_ratio:.1f}%)\n"
                    f"Crash: `{row['crash_value']}x`",
                    title="💰 WHALE PAYOUT"
                )
                
            # Velocity Anomaly Check
            vm_str = row.get('velocity_metrics', '[]')
            vm = json.loads(vm_str)
            if vm:
                deltas = [entry['delta_ms'] for entry in vm if 'delta_ms' in entry]
                if len(deltas) > 5:
                    avg_delta = sum(deltas[:-1]) / (len(deltas)-1)
                    last_delta = deltas[-1]
                    if last_delta < avg_delta * 0.5: # 2x speedup
                        send_alert(
                            f"⚠️ **SPEED MANIPULATION DETECTED!**\n"
                            f"Multiplier accelerated by 2x just before crash.\n"
                            f"Avg Step: `{avg_delta:.0f}ms` | Last Step: `{last_delta:.0f}ms`",
                            title="⚠️ VELOCITY ANOMALY"
                        )
        except Exception as e:
            print(f"Alert logic error: {e}")

        print(f"⏱️ Round {data.get('roundId')} data saved: {data.get('durationMs')}ms, {data.get('crashValue')}x")

        return jsonify({'success': True})
    except Exception as e:
        print(f"⚠️ Error saving round data: {e}")
        return jsonify({'error': str(e)}), 400



# Store latest betting behavior data
betting_behavior = {
    'totalBettors': 0,
    'totalStaked': 0,
    'cashedOutCount': 0,
    'stillBettingCount': 0,
    'poolSize': 0,
    'onlinePlayers': 0,
    'balance': 0,  # User's current balance
    'liveCoefficient': 0,  # Current live multiplier
    'roundId': None,
    'lastCoefficient': None,
    'history': []  # Keep last 50 snapshots
}

@server.route('/api/betting-behavior', methods=['POST', 'OPTIONS'])
def api_betting_behavior():
    global betting_behavior, last_ip
    if request.method == 'OPTIONS': return '', 200
    try:
        # ProxyFix handles X-Forwarded-For automatically
        real_ip = request.remote_addr
        last_ip = real_ip
        
        data = request.get_json()
        
        # LATENCY TRACKING: Calculate delay from extension to server
        ext_timestamp = data.get('timestamp', 0)
        if ext_timestamp:
            server_time = int(time.time() * 1000)  # Current server time in ms
            latency_ms = server_time - ext_timestamp
            print(f"📡 LATENCY: {latency_ms}ms (Ext→Server) | Bettors: {data.get('totalBettors', 0)}")
        
        # Update live stats
        betting_behavior['totalBettors'] = data.get('totalBettors', 0)
        betting_behavior['totalStaked'] = data.get('totalStaked', 0)
        betting_behavior['cashedOutCount'] = len(data.get('cashedOut', []))
        betting_behavior['stillBettingCount'] = len(data.get('stillBetting', []))
        betting_behavior['poolSize'] = data.get('poolSize', 0)
        betting_behavior['onlinePlayers'] = data.get('onlinePlayers', 0)
        betting_behavior['liveCoefficient'] = data.get('liveCoefficient', 0)
        betting_behavior['timestamp'] = data.get('timestamp')
        
        # IMPORTANT: Store raw active bets for Whale Radar
        betting_behavior['activeBets'] = data.get('activeBets', [])

        
        # Update round info
        if data.get('roundId'):
            betting_behavior['roundId'] = data.get('roundId')
        if data.get('lastCoefficient'):
            betting_behavior['lastCoefficient'] = data.get('lastCoefficient')
        
        # Update balance for risk management
        if data.get('balance', 0) > 0:
            betting_behavior['balance'] = data.get('balance')
            # Update predictor's bankroll for Kelly calculations
            if ADVANCED_PREDICTOR:
                predictor.bankroll = data.get('balance')
            
        # FEED PREDICTOR
        if ADVANCED_PREDICTOR:
            predictor.add_behavioral_data(betting_behavior)
        
        # Store in history for pattern analysis
        snapshot = {
            'timestamp': data.get('timestamp'),
            'bettors': betting_behavior['totalBettors'],
            'staked': betting_behavior['totalStaked'],
            'cashedOut': betting_behavior['cashedOutCount'],
            'stillIn': betting_behavior['stillBettingCount'],
            'pool': betting_behavior['poolSize'],
            'roundId': betting_behavior['roundId']
        }
        betting_behavior['history'].append(snapshot)
        
        # Keep only last 50 snapshots
        if len(betting_behavior['history']) > 50:
            betting_behavior['history'] = betting_behavior['history'][-50:]
        
        # Log with round context
        if betting_behavior['totalBettors'] > 0:
            cashout_ratio = betting_behavior['cashedOutCount'] / betting_behavior['totalBettors']
            round_str = f"R{betting_behavior['roundId']}" if betting_behavior['roundId'] else "?"
            print(f"📊 {round_str}: {betting_behavior['totalBettors']} bettors, {cashout_ratio*100:.0f}% out, pool: {betting_behavior['poolSize']:,}")
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@server.route('/api/betting-behavior', methods=['GET'])
def get_betting_behavior():
    return jsonify(betting_behavior)

@server.route('/api/verify', methods=['POST'])
def api_verify():
    """Verify a round hash/key pair"""
    try:
        data = request.get_json()
        key = data.get('key', '')
        r_hash = data.get('hash', '')
        
        if not ProvablyFair:
            return jsonify({'error': 'Provably Fair module not loaded'}), 500
            
        calculated_hash = ProvablyFair.generate_hash(key)
        is_valid = calculated_hash == r_hash
        multiplier = ProvablyFair.calculate_multiplier(key)
        serial = ProvablyFair.extract_serial(key)
        
        return jsonify({
            'success': True,
            'valid': is_valid,
            'calculated_hash': calculated_hash,
            'multiplier': multiplier,
            'serial': serial
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400



# ============ PREMIUM STYLES ============
# Styles are now loaded from assets/tactical_os.css to enable proper layout control
STYLES = ""

# ============ LAYOUT ============
app.index_string = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
</head>
<body>
    ''' + STYLES + '''
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>'''

def create_sidebar():
    return html.Div([
        # Logo Section
        html.Div([
            html.Div([
                html.Span("◆", style={'fontSize': '28px', 'marginRight': '12px', 'color': '#d4af37'}),
                html.Div([
                    html.Div("ZEPPELIN", style={'fontSize': '18px', 'fontWeight': '900', 'letterSpacing': '3px', 'color': '#fff'}),
                    html.Div("V2", style={'fontSize': '10px', 'fontWeight': '600', 'letterSpacing': '4px', 'color': 'rgba(255,255,255,0.3)'})
                ])
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={'padding': '0 0 40px 0', 'borderBottom': '1px solid rgba(255,255,255,0.05)', 'marginBottom': '24px'}),

        # Simplified Navigation
        html.Div([
            dcc.Link("Dashboard", href="/", className="nav-link", id="link-terminal", 
                     style={'display': 'block', 'padding': '12px 0', 'color': '#d4af37', 'fontWeight': '700', 'letterSpacing': '1px'}),
            html.Div(style={'height': '1px', 'background': 'rgba(255,255,255,0.05)', 'margin': '12px 0'}),
            dcc.Link("Analytics", href="/charts", className="nav-link", id="link-charts",
                     style={'display': 'block', 'padding': '8px 0', 'color': 'rgba(255,255,255,0.4)', 'fontSize': '13px'}),
            dcc.Link("Temporal", href="/temporal", className="nav-link", id="link-temporal",
                     style={'display': 'block', 'padding': '8px 0', 'color': 'rgba(255,255,255,0.4)', 'fontSize': '13px'}),
            dcc.Link("Audit", href="/audit", className="nav-link", id="link-audit",
                     style={'display': 'block', 'padding': '8px 0', 'color': 'rgba(255,255,255,0.4)', 'fontSize': '13px'}),
        ], style={'flex': 1}),

        # Status
        html.Div([
            html.Div([
                html.Div("SYSTEM", style={'fontSize': '9px', 'letterSpacing': '2px', 'color': 'rgba(255,255,255,0.3)', 'marginBottom': '8px'}),
                html.Div([
                    html.Span("●", style={'color': '#d4af37', 'marginRight': '8px'}),
                    html.Span("Online", style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.5)'})
                ])
            ], style={'padding': '16px 0', 'borderTop': '1px solid rgba(255,255,255,0.05)'})
        ]),
        
        # Data display
        html.Div(id="sidebar-crashes", style={'display': 'none'}),
        
        # Language Toggle
        html.Div([
            html.Div("LANGUAGE / LUGHA", style={'fontSize': '9px', 'letterSpacing': '2px', 'color': 'rgba(255,255,255,0.3)', 'marginBottom': '8px'}),
            html.Div([
                html.Span("EN", id="lang-en", n_clicks=0, style={'marginRight': '10px', 'cursor': 'pointer', 'fontWeight': 'bold', 'color': 'var(--kinetic-cyan)'}),
                html.Span("|", style={'color': 'rgba(255,255,255,0.2)', 'marginRight': '10px'}),
                html.Span("SW", id="lang-sw", n_clicks=0, style={'cursor': 'pointer', 'color': 'rgba(255,255,255,0.5)'})
            ])
        ], style={'padding': '16px 0', 'borderTop': '1px solid rgba(255,255,255,0.05)'})
    ], className="sidebar")



def create_dashboard():
    """V4 Tactical OS - High-Density Intelligence Console"""
    return html.Div([
        
        # 0. HISTORY TAPE (Pinned across top)
        html.Div(id="history-strip", className="history-strip", style={'borderBottom': '1px solid var(--hud-border)'}),

        # ═══════════════════════════════════════════════════════════════
        # HUD GRID SYSTEM (Modular Intelligence Layer)
        # ═══════════════════════════════════════════════════════════════
        html.Div([
            
            # MODULE 1: COMMAND STATUS (Cols 1-2)
            html.Div([
                html.Span("COMMAND STATUS", className="hud-label", id="hdr-command"),
                dbc.Tooltip("Loading...", target="hdr-command", id="tip-command", placement="bottom"),
                html.Div([
                    html.Div(id="signal-text", children="WAIT", className="mono", style={'fontSize': '32px', 'fontWeight': '900'}),
                    html.Div([
                        html.Span("LOAD ALLOC: ", id="lbl-load-alloc", style={'color': 'rgba(255,255,255,0.3)'}),
                        html.Span(id="signal-info", children="TZS 0", className="cyan", style={'fontWeight': '700'})
                    ], style={'marginTop': '2px', 'fontSize': '11px'})
                ])
            ], className="hud-module", style={'gridColumn': 'span 2'}),
            
            # MODULE 2: CONSENSUS ENGINE (Cols 3-4)
            html.Div([
                html.Span("CONSENSUS ENGINE", className="hud-label", id="hdr-consensus"),
                dbc.Tooltip("Loading...", target="hdr-consensus", id="tip-consensus", placement="bottom"),
                html.Div([
                    html.Div(id="consensus-pct", children="0%", className="hud-value", style={'fontSize': '28px'}),
                    html.Div([
                        html.Span("OPTIMAL TARGET: ", id="lbl-optimal", style={'color': 'rgba(255,255,255,0.3)'}),
                        html.Span(id="stat-target", children="2.0x", className="cyan")
                    ], style={'marginTop': '2px', 'fontSize': '11px'})
                ])
            ], className="hud-module", style={'gridColumn': 'span 2'}),
            
            # MODULE 3: SESSION ANALYTICS (Cols 5-12)
            html.Div([
                html.Span("SESSION ANALYTICS", className="hud-label", id="hdr-session"),
                html.Div([
                    html.Div([
                        html.Div("WIN RATE", id="lbl-winrate", style={'fontSize': '9px', 'color': 'rgba(255,255,255,0.4)'}),
                        html.Div(id="stat-winrate", children="0%", className="mono cyan", style={'fontSize': '18px'})
                    ], style={'flex': 1}),
                    html.Div([
                        html.Div("SESSION PROFIT", id="lbl-profit", style={'fontSize': '9px', 'color': 'rgba(255,255,255,0.4)'}),
                        html.Div(id='bet-balance', children="TZS 0", className="mono", style={'fontSize': '18px'})
                    ], style={'flex': 1.5}),
                    html.Div([
                        html.Div("TOTAL LOGS", id="lbl-total-logs", style={'fontSize': '9px', 'color': 'rgba(255,255,255,0.4)'}),
                        html.Div(id="sidebar-crashes-val", children="0", className="mono", style={'fontSize': '18px'})
                    ], style={'flex': 1}),
                ], style={'display': 'flex', 'marginTop': '8px'})
            ], className="hud-module", style={'gridColumn': 'span 8'}),
            
            # MODULE 4: INTELLIGENCE STRIP (Vertical - Col 1-2)
            html.Div([
                # POF (Pool Overflow Factor)
                html.Div([
                    html.Span("POOL OVERFLOW", className="hud-label", id="lbl-pool-load"),
                    dbc.Tooltip("Loading...", target="lbl-pool-load", id="tip-pof", placement="right"),
                    html.Div(id="metric-pof", children="1.0x", className="mono", style={'fontSize': '20px'})
                ], style={'paddingBottom': '10px', 'borderBottom': '1px solid var(--hud-border)'}),
                
                # VELOCITY
                html.Div([
                    html.Span("FLOW VELOCITY", className="hud-label", style={'marginTop': '10px'}, id="lbl-velocity"),
                    html.Div(id="metric-mv", children="NORMAL", className="mono", style={'fontSize': '14px'})
                ], style={'paddingBottom': '10px', 'borderBottom': '1px solid var(--hud-border)'}),

                # MODEL HEALTH
                html.Div([
                    html.Span("ACTIVE MODELS", className="hud-label", style={'marginTop': '10px'}, id="lbl-active-models"),
                    html.Div(id="model-health-display", style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px', 'marginBottom': '4px'})
                ], style={'paddingBottom': '10px', 'borderBottom': '1px solid var(--hud-border)'}),
                
                # PROBABILITIES
                html.Div([
                    html.Span("EDGE ESTIMATE", className="hud-label", style={'marginTop': '10px'}, id="lbl-edge"),
                    html.Div([
                        html.Div([html.Span("1.5x: ", style={'fontSize': '9px', 'color': 'rgba(255,255,255,0.3)'}), html.Span("68%", className="cyan")], style={'fontSize': '11px'}),
                        html.Div([html.Span("2.0x: ", style={'fontSize': '9px', 'color': 'rgba(255,255,255,0.3)'}), html.Span("42%")], style={'fontSize': '11px'}),
                        html.Div([html.Span("3.0x: ", style={'fontSize': '9px', 'color': 'rgba(255,255,255,0.3)'}), html.Span("18%")], style={'fontSize': '11px'}),
                    ])
                ])
            ], className="hud-module", style={'gridColumn': 'span 2', 'gridRow': 'span 2'}),
            
            # MODULE 5: PERFORMANCE VECTOR (Cols 3-9)
            html.Div([
                html.Span("PERFORMANCE VECTOR", className="hud-label", id="hdr-perf"),
                dbc.Tooltip("Loading...", target="hdr-perf", id="tip-perf", placement="bottom"),
                dcc.Graph(id="main-chart", config={'displayModeBar': False}, 
                         style={'height': '360px', 'background': 'transparent'})
            ], className="hud-module", style={'gridColumn': 'span 7', 'gridRow': 'span 2'}),
            
            # MODULE 6: WHALE RADAR (Cols 10-12)
            html.Div([
                html.Span("WHALE RADAR", className="hud-label", id="hdr-whale"),
                dbc.Tooltip("Loading...", target="hdr-whale", id="tip-whale", placement="bottom"),
                dcc.Graph(id="whale-chart", config={'displayModeBar': False}, 
                         style={'height': '360px', 'background': 'transparent'})
            ], className="hud-module", style={'gridColumn': 'span 3', 'gridRow': 'span 2'}),

            # MODULE 7: LOG STREAM (Full Width Bottom)
            html.Div([
                html.Span("TERMINAL LOG STREAM", className="hud-label", id="lbl-log"),
                dbc.Tooltip("Loading...", target="lbl-log", id="tip-log", placement="top"),
                html.Div(id="logic-scoreboard", style={
                    'fontSize': '10px', 'color': 'rgba(255,255,255,0.5)', 
                    'overflowY': 'auto', 'height': '120px', 'fontFamily': 'var(--mono-font)'
                })
            ], className="hud-module", style={'gridColumn': 'span 8'}),

            # MODULE 8: STRATEGIC INTERPRETER (New Narrative Module)
            html.Div([
                html.Span("STRATEGIC INTERPRETER", className="hud-label", id="lbl-interpreter"),
                html.Div(id="narrative-display", style={
                    'fontSize': '14px', 'color': '#fff', 'padding': '10px',
                    'lineHeight': '1.4', 'fontFamily': 'var(--mono-font)',
                    'borderLeft': '2px solid var(--kinetic-cyan)',
                    'background': 'rgba(0, 240, 255, 0.05)'
                })
            ], className="hud-module", style={'gridColumn': 'span 4'}),

        ], className="hud-grid"),


        # Hidden elements for state tracking
        html.Div(id="bet-online", style={'display': 'none'}),
        html.Div(id="last-crash", style={'display': 'none'}),
        html.Div(id="last-update", style={'display': 'none'}),
        # stat-winrate and bet-balance are already present in HUD modules
        html.Div(id="stat-regime", style={'display': 'none'}),
        html.Div(id="stat-volatility", style={'display': 'none'}),
        html.Div(id="last-5-rounds", style={'display': 'none'}),
        html.Div(id="current-round", style={'display': 'none'}),
        html.Div(id="bet-livecoef", style={'display': 'none'}),
        
    ], style={'display': 'flex', 'flexDirection': 'column', 'flex': 1})

# ============ TRANSLATION CALLBACKS ============

@app.callback(
    [Output('language-store', 'data'),
     Output('lang-en', 'style'),
     Output('lang-sw', 'style')],
    [Input('lang-en', 'n_clicks'),
     Input('lang-sw', 'n_clicks')],
    [State('language-store', 'data')]
)
def switch_language(n_en, n_sw, current_lang):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_lang, {}, {}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    new_lang = 'en' if button_id == 'lang-en' else 'sw'
    
    style_active = {'color': 'var(--kinetic-cyan)', 'fontWeight': 'bold', 'cursor': 'pointer'}
    style_inactive = {'color': 'rgba(255,255,255,0.5)', 'cursor': 'pointer'}
    
    return (
        new_lang,
        style_active if new_lang == 'en' else style_inactive,
        style_active if new_lang == 'sw' else style_inactive
    )

@app.callback(
    [Output('link-terminal', 'children'),
     Output('link-charts', 'children'),
     Output('link-temporal', 'children'),
     Output('link-audit', 'children'),
     Output('lbl-regime-status', 'children'),
     Output('lbl-sys-latency', 'children'),
     Output('lbl-target', 'children'),
     Output('lbl-consensus', 'children')],
    [Input('language-store', 'data')]
)
def update_nav_labels(lang):
    l = lang if lang in TRANSLATIONS else 'en'
    t = TRANSLATIONS[l]
    return (
        t['nav_dashboard'], t['nav_analytics'], t['nav_temporal'], t['nav_audit'],
        t['lbl_regime_status'], t['lbl_sys_latency'], t['lbl_target'], t['lbl_consensus']
    )

# Dashboard-specific translations (only fire on dashboard page)
@app.callback(
    [Output('hdr-command', 'children'),
     Output('hdr-consensus', 'children'),
     Output('hdr-session', 'children'),
     Output('lbl-pool-load', 'children'),
     Output('lbl-velocity', 'children'),
     Output('lbl-active-models', 'children'),
     Output('lbl-log', 'children'),
     Output('lbl-interpreter', 'children'),
     Output('hdr-perf', 'children'),
     Output('hdr-whale', 'children'),
     Output('lbl-winrate', 'children'),
     Output('lbl-profit', 'children'),
     Output('lbl-total-logs', 'children'),
     Output('lbl-optimal', 'children'),
     Output('lbl-edge', 'children'),
     Output('lbl-load-alloc', 'children'),
     Output('tip-command', 'children'),
     Output('tip-consensus', 'children'),
     Output('tip-whale', 'children'),
     Output('tip-perf', 'children'),
     Output('tip-log', 'children'),
     Output('tip-pof', 'children')],
    [Input('language-store', 'data'),
     Input('url', 'pathname')]
)
def update_dashboard_labels(lang, pathname):
    if pathname != '/':
        raise dash.exceptions.PreventUpdate
    l = lang if lang in TRANSLATIONS else 'en'
    t = TRANSLATIONS[l]
    return (
        t['mod_command'], t['mod_consensus'], t['mod_session'],
        t['lbl_pool_load'], t['lbl_velocity'], t['lbl_active_models'],
        t['mod_log'], t['mod_interpreter'], t['mod_perf'], t['mod_whale'],
        t['lbl_winrate'], t['lbl_profit'], t['lbl_total_logs'],
        t['lbl_optimal'], t['lbl_edge'], t['lbl_load_alloc'],
        t['tip_command'], t['tip_consensus'], t['tip_whale'],
        t['tip_perf'], t['tip_log'], t['tip_pof']
    )

# Charts page translations
@app.callback(
    [Output('page-charts-title', 'children'),
     Output('lbl-crash-mult', 'children'),
     Output('lbl-chart-desc', 'children'),
     Output('lbl-winrate-trend', 'children'),
     Output('lbl-volatility', 'children'),
     Output('lbl-distribution', 'children')],
    [Input('language-store', 'data'),
     Input('url', 'pathname')]
)
def update_charts_labels(lang, pathname):
    if pathname != '/charts':
        raise dash.exceptions.PreventUpdate
    l = lang if lang in TRANSLATIONS else 'en'
    t = TRANSLATIONS[l]
    return (
        t['page_charts_title'], t['lbl_crash_mult'], t['lbl_chart_desc'],
        t['lbl_winrate_trend'], t['lbl_volatility'], t['lbl_distribution']
    )

# Audit page translations
@app.callback(
    [Output('page-audit-title', 'children'),
     Output('page-audit-desc', 'children'),
     Output('lbl-house-pl', 'children'),
     Output('lbl-velocity-audit', 'children'),
     Output('lbl-deep-audit', 'children'),
     Output('lbl-audit-desc', 'children')],
    [Input('language-store', 'data'),
     Input('url', 'pathname')]
)
def update_audit_labels(lang, pathname):
    if pathname != '/audit':
        raise dash.exceptions.PreventUpdate
    l = lang if lang in TRANSLATIONS else 'en'
    t = TRANSLATIONS[l]
    return (
        t['page_audit_title'], t['page_audit_desc'], t['lbl_house_pl'],
        t['lbl_velocity_audit'], t['lbl_deep_audit'], t['lbl_audit_desc']
    )

# Temporal page translations
@app.callback(
    [Output('page-temporal-title', 'children'),
     Output('page-temporal-desc', 'children'),
     Output('lbl-timeframe', 'children')],
    [Input('language-store', 'data'),
     Input('url', 'pathname')]
)
def update_temporal_labels(lang, pathname):
    if pathname != '/temporal':
        raise dash.exceptions.PreventUpdate
    l = lang if lang in TRANSLATIONS else 'en'
    t = TRANSLATIONS[l]
    return (
        t['page_temporal_title'], t['page_temporal_desc'], t['lbl_timeframe']
    )

def create_charts_page():
    return html.Div([
        html.H1("Charts", id="page-charts-title", style={'fontSize': '28px', 'fontWeight': '700', 'marginBottom': '24px'}),
        
        # Main Chart
        html.Div([
            html.Div([
                html.Span("📈 Crash Multiplier", id="lbl-crash-mult", style={'fontWeight': '600'}),
                html.Span(" • With MA10, MA30 & Bollinger Bands", id="lbl-chart-desc", style={'color': 'rgba(255,255,255,0.4)', 'fontSize': '12px'})
            ], style={'marginBottom': '16px'}),
            dcc.Graph(id="pro-chart", config={'displayModeBar': True}, style={'height': '400px'})
        ], className="glass-card", style={'padding': '24px', 'marginBottom': '20px'}),
        
        # Sub Charts
        html.Div([
            html.Div([
                html.Div("Win Rate Trend", id="lbl-winrate-trend", style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.5)', 'marginBottom': '8px'}),
                dcc.Graph(id="winrate-chart", config={'displayModeBar': False}, style={'height': '160px'})
            ], className="glass-card", style={'padding': '16px', 'flex': 1, 'marginRight': '16px'}),
            html.Div([
                html.Div("Volatility", id="lbl-volatility", style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.5)', 'marginBottom': '8px'}),
                dcc.Graph(id="vol-chart", config={'displayModeBar': False}, style={'height': '160px'})
            ], className="glass-card", style={'padding': '16px', 'flex': 1, 'marginRight': '16px'}),
            html.Div([
                html.Div("Distribution", id="lbl-distribution", style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.5)', 'marginBottom': '8px'}),
                dcc.Graph(id="dist-chart", config={'displayModeBar': False}, style={'height': '160px'})
            ], className="glass-card", style={'padding': '16px', 'flex': 1}),
        ], style={'display': 'flex'})
    ], style={'padding': '32px'})


def create_analysis_page():
    return html.Div([
        html.H1("Analysis", style={'fontSize': '28px', 'fontWeight': '700', 'marginBottom': '24px'}),
        
        html.Div([
            html.Div([
                html.Div("🎯 Trend", style={'fontWeight': '600', 'marginBottom': '12px'}),
                html.Div(id="a-trend")
            ], className="glass-card", style={'padding': '20px', 'flex': 1, 'marginRight': '16px', 'marginBottom': '16px'}),
            html.Div([
                html.Div("🔄 Pattern", style={'fontWeight': '600', 'marginBottom': '12px'}),
                html.Div(id="a-pattern")
            ], className="glass-card", style={'padding': '20px', 'flex': 1, 'marginBottom': '16px'}),
        ], style={'display': 'flex'}),
        
        html.Div([
            html.Div([
                html.Div("📊 Regime", style={'fontWeight': '600', 'marginBottom': '12px'}),
                html.Div(id="a-regime")
            ], className="glass-card", style={'padding': '20px', 'flex': 1, 'marginRight': '16px'}),
            html.Div([
                html.Div("👥 Behavior", style={'fontWeight': '600', 'marginBottom': '12px'}),
                html.Div(id="a-behavior")
            ], className="glass-card", style={'padding': '20px', 'flex': 1}),
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        html.Div([
            html.Div("🎲 Probability Matrix", style={'fontWeight': '600', 'marginBottom': '16px'}),
            dcc.Graph(id="prob-matrix", config={'displayModeBar': False}, style={'height': '250px'})
        ], className="glass-card", style={'padding': '24px'})
    ], style={'padding': '32px'})
def create_audit_page():
    return html.Div([
        html.Div([
            html.H1("Audit Hub", id="page-audit-title", style={'fontSize': '32px', 'fontWeight': '800', 'margin': 0}),
            html.Div("🛡️ Real-time house integrity and payout auditing", id="page-audit-desc", style={'color': 'rgba(255,255,255,0.4)', 'fontSize': '14px', 'marginTop': '4px'})
        ], style={'marginBottom': '40px'}),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("House P&L (Recent Rounds)", id="lbl-house-pl", style={'fontSize': '14px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.4)', 'marginBottom': '20px'}),
                    dcc.Graph(id="audit-pl-chart", config={'displayModeBar': False}, style={'height': '220px'})
                ], className="glass-card", style={'padding': '24px', 'height': '100%'})
            ], width=7),
            dbc.Col([
                html.Div([
                    html.H4("Velocity Audit (ms/step)", id="lbl-velocity-audit", style={'fontSize': '14px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.4)', 'marginBottom': '20px'}),
                    dcc.Graph(id="audit-velocity-chart", config={'displayModeBar': False}, style={'height': '220px'})
                ], className="glass-card", style={'padding': '24px', 'height': '100%'})
            ], width=5),
        ], style={'marginBottom': '30px'}),
        
        html.Div([
            html.Div([
                html.Span("🔍 Deep Audit Table", id="lbl-deep-audit", style={'fontWeight': '700'}),
                html.Span(" • Last 20 Rounds with Seed Verification", id="lbl-audit-desc", style={'color': 'rgba(255,255,255,0.3)', 'fontSize': '12px', 'marginLeft': '10px'})
            ], style={'marginBottom': '20px'}),
            html.Div(id="audit-stats-table")
        ], className="glass-card", style={'padding': '30px'})
    ], style={'padding': '32px'})


def create_history_page():
    return html.Div([
        html.Div([
            html.H1("History", style={'fontSize': '28px', 'fontWeight': '700', 'margin': 0}),
            dbc.Button("Clear All", id="clear-btn", color="danger", size="sm", 
                      style={'borderRadius': '8px', 'background': 'linear-gradient(135deg, #ef4444, #dc2626)', 'border': 'none'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '24px'}),
        
        # Stats Row
        html.Div([
            html.Div([html.Div(id="h-total", style={'fontSize': '24px', 'fontWeight': '700', 'color': '#10b981'}),
                     html.Div("Total", style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '11px'})], 
                    className="glass-card", style={'padding': '16px', 'textAlign': 'center', 'flex': 1, 'marginRight': '12px'}),
            html.Div([html.Div(id="h-avg", style={'fontSize': '24px', 'fontWeight': '700', 'color': '#3b82f6'}),
                     html.Div("Average", style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '11px'})], 
                    className="glass-card", style={'padding': '16px', 'textAlign': 'center', 'flex': 1, 'marginRight': '12px'}),
            html.Div([html.Div(id="h-max", style={'fontSize': '24px', 'fontWeight': '700', 'color': '#10b981'}),
                     html.Div("Highest", style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '11px'})], 
                    className="glass-card", style={'padding': '16px', 'textAlign': 'center', 'flex': 1, 'marginRight': '12px'}),
            html.Div([html.Div(id="h-min", style={'fontSize': '24px', 'fontWeight': '700', 'color': '#ef4444'}),
                     html.Div("Lowest", style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '11px'})], 
                    className="glass-card", style={'padding': '16px', 'textAlign': 'center', 'flex': 1}),
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        # History Grid
        html.Div([
            html.Div(id="history-grid", style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px', 'maxHeight': '400px', 'overflowY': 'auto'})
        ], className="glass-card", style={'padding': '20px'})
    ], style={'padding': '32px'})


def create_temporal_page():
    return html.Div([
        # Page Header
        html.Div([
            html.H1("🕐 Time Myths & Temporal Intelligence", id="page-temporal-title", style={'fontSize': '32px', 'fontWeight': '800', 'margin': 0, 'color': '#fff'}),
            html.Div("Real-time insights from temporal pattern analysis", id="page-temporal-desc", style={'color': 'rgba(255,255,255,0.4)', 'fontSize': '14px', 'marginTop': '4px'})
        ], style={'marginBottom': '30px'}),
        
        # Timeframe Selector
        html.Div([
            html.Span("Analysis Timeframe:", id="lbl-timeframe", style={'color': 'rgba(255,255,255,0.6)', 'marginRight': '20px', 'fontSize': '14px', 'fontWeight': '600'}),
            dbc.RadioItems(
                id="temporal-timeframe-selector",
                options=[
                    {"label": "All Time", "value": "all"},
                    {"label": "Last 24 Hours", "value": "24h"},
                    {"label": "Last 7 Days", "value": "7d"},
                    {"label": "Weight Recent Data", "value": "weighted"},
                ],
                value="all",
                inline=True,
                style={'display': 'inline-block'},
                labelStyle={'marginRight': '20px', 'color': '#fff', 'fontSize': '14px'},
                inputStyle={'marginRight': '8px'}
            )
        ], className="glass-card", style={'padding': '15px 25px', 'marginBottom': '24px', 'display': 'flex', 'alignItems': 'center'}),
        
        # TOP ALERT BANNER - Current Time Status
        html.Div(id="temporal-alert-banner", className="glass-card", style={'padding': '20px', 'marginBottom': '24px'}),
        
        # ROW 1: Best/Worst Hours + Risk Indicators
        dbc.Row([
            # Best Hour Card
            dbc.Col([
                html.Div([
                    html.Div("🏆 BEST HOUR", style={'fontSize': '11px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.4)', 'letterSpacing': '2px', 'marginBottom': '10px'}),
                    html.Div(id="best-hour-display", style={'fontSize': '36px', 'fontWeight': '800', 'color': '#10b981'}),
                    html.Div(id="best-hour-avg", style={'fontSize': '14px', 'color': 'rgba(255,255,255,0.6)', 'marginTop': '5px'})
                ], className="glass-card", style={'padding': '24px', 'textAlign': 'center', 'background': 'linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.02))', 'border': '1px solid rgba(16, 185, 129, 0.3)'})
            ], width=3),
            
            # Worst Hour Card
            dbc.Col([
                html.Div([
                    html.Div("🚫 WORST HOUR", style={'fontSize': '11px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.4)', 'letterSpacing': '2px', 'marginBottom': '10px'}),
                    html.Div(id="worst-hour-display", style={'fontSize': '36px', 'fontWeight': '800', 'color': '#ef4444'}),
                    html.Div(id="worst-hour-avg", style={'fontSize': '14px', 'color': 'rgba(255,255,255,0.6)', 'marginTop': '5px'})
                ], className="glass-card", style={'padding': '24px', 'textAlign': 'center', 'background': 'linear-gradient(145deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.02))', 'border': '1px solid rgba(239, 68, 68, 0.3)'})
            ], width=3),
            
            # Instant Crash Risk
            dbc.Col([
                html.Div([
                    html.Div("💥 INSTANT CRASH RISK", style={'fontSize': '11px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.4)', 'letterSpacing': '2px', 'marginBottom': '10px'}),
                    html.Div(id="instant-crash-rate", style={'fontSize': '36px', 'fontWeight': '800', 'color': '#f59e0b'}),
                    html.Div("Current hour ≤1.1x rate", style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.5)', 'marginTop': '5px'})
                ], className="glass-card", style={'padding': '24px', 'textAlign': 'center'})
            ], width=3),
            
            # Velocity Anomalies
            dbc.Col([
                html.Div([
                    html.Div("⚡ SPEED ANOMALIES", style={'fontSize': '11px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.4)', 'letterSpacing': '2px', 'marginBottom': '10px'}),
                    html.Div(id="velocity-anomaly-count", style={'fontSize': '36px', 'fontWeight': '800', 'color': '#8b5cf6'}),
                    html.Div("Last 100 rounds", style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.5)', 'marginTop': '5px'})
                ], className="glass-card", style={'padding': '24px', 'textAlign': 'center'})
            ], width=3),
        ], style={'marginBottom': '24px'}),
        
        # ROW 2: High Stake Warning (conditional)
        html.Div(id="high-stake-warning-container"),
        
        # ROW 3: Heatmap + Intel Panel
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("📊 Hourly Performance Heatmap", style={'fontSize': '16px', 'fontWeight': '700', 'marginBottom': '20px', 'color': 'rgba(255,255,255,0.6)'}),
                    dcc.Graph(id="temporal-heatmap", style={'height': '350px'})
                ], className="glass-card", style={'padding': '30px'})
            ], width=8),
            dbc.Col([
                html.Div([
                    html.H4("🎯 Strategic Intel", style={'fontSize': '16px', 'fontWeight': '700', 'marginBottom': '20px', 'color': 'rgba(255,255,255,0.6)'}),
                    html.Div(id="temporal-intel-content")
                ], className="glass-card", style={'padding': '30px', 'height': '100%'})
            ], width=4)
        ], style={'marginBottom': '24px'}),
        
        # ROW 4: Hourly Chart + Velocity Anomaly Details
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("📈 Hourly Crash Averages", style={'fontSize': '16px', 'fontWeight': '700', 'marginBottom': '20px', 'color': 'rgba(255,255,255,0.6)'}),
                    dcc.Graph(id="hourly-cycle-chart", style={'height': '280px'})
                ], className="glass-card", style={'padding': '30px'})
            ], width=8),
            dbc.Col([
                html.Div([
                    html.H4("🚨 Recent Anomalies", style={'fontSize': '16px', 'fontWeight': '700', 'marginBottom': '20px', 'color': 'rgba(255,255,255,0.6)'}),
                    html.Div(id="velocity-anomaly-list")
                ], className="glass-card", style={'padding': '30px', 'height': '100%'})
            ], width=4)
        ])
    ], style={'padding': '32px'})


def create_betting_window_page():
    """
    V2 Premium Betting Windows Page
    Monochromatic black/white theme with gold accents
    """
    return html.Div([
        # V2 Premium Header
        html.Div([
            html.H1("BETTING WINDOWS", style={
                'fontSize': '48px', 'fontWeight': '900', 'margin': 0, 'color': '#fff',
                'letterSpacing': '8px', 'textAlign': 'center'
            }),
            html.Div("Regime Detection Intelligence • V2", style={
                'color': 'rgba(255,255,255,0.4)', 'fontSize': '12px', 'textAlign': 'center',
                'letterSpacing': '4px', 'marginTop': '8px'
            })
        ], style={'marginBottom': '48px', 'paddingTop': '20px'}),
        
        # WINDOW STATUS BANNER
        html.Div(id="window-status-banner", style={
            'background': '#000',
            'border': '2px solid rgba(255,255,255,0.1)',
            'borderRadius': '0',
            'padding': '48px',
            'textAlign': 'center',
            'marginBottom': '48px'
        }),
        
        # ROW 1: Key Metrics (Monochrome)
        dbc.Row([
            # Consensus Score
            dbc.Col([
                html.Div([
                    html.Div("CONSENSUS SCORE", style={
                        'fontSize': '10px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.3)',
                        'letterSpacing': '3px', 'marginBottom': '16px'
                    }),
                    html.Div(id="window-consensus-score", style={
                        'fontSize': '64px', 'fontWeight': '900', 'color': '#fff'
                    }),
                    html.Div("S_w Formula Output", style={
                        'fontSize': '10px', 'color': 'rgba(255,255,255,0.3)', 'marginTop': '8px'
                    })
                ], style={
                    'background': '#0a0a0a', 'border': '1px solid rgba(255,255,255,0.08)',
                    'padding': '40px', 'textAlign': 'center'
                })
            ], width=3),
            
            # Current Regime
            dbc.Col([
                html.Div([
                    html.Div("CURRENT STATE", style={
                        'fontSize': '10px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.3)',
                        'letterSpacing': '3px', 'marginBottom': '16px'
                    }),
                    html.Div(id="window-current-state", style={
                        'fontSize': '24px', 'fontWeight': '800', 'color': '#fff'
                    }),
                    html.Div(id="window-state-prob", style={
                        'fontSize': '12px', 'color': 'rgba(255,255,255,0.4)', 'marginTop': '8px'
                    })
                ], style={
                    'background': '#0a0a0a', 'border': '1px solid rgba(255,255,255,0.08)',
                    'padding': '40px', 'textAlign': 'center'
                })
            ], width=3),
            
            # POF Status
            dbc.Col([
                html.Div([
                    html.Div("POOL OVERLOAD", style={
                        'fontSize': '10px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.3)',
                        'letterSpacing': '3px', 'marginBottom': '16px'
                    }),
                    html.Div(id="window-pof-value", style={
                        'fontSize': '48px', 'fontWeight': '900', 'color': '#fff'
                    }),
                    html.Div("POF Ratio", style={
                        'fontSize': '10px', 'color': 'rgba(255,255,255,0.3)', 'marginTop': '8px'
                    })
                ], style={
                    'background': '#0a0a0a', 'border': '1px solid rgba(255,255,255,0.08)',
                    'padding': '40px', 'textAlign': 'center'
                })
            ], width=3),
            
            # Kelly Recommendation
            dbc.Col([
                html.Div([
                    html.Div("KELLY BET", style={
                        'fontSize': '10px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.3)',
                        'letterSpacing': '3px', 'marginBottom': '16px'
                    }),
                    html.Div(id="window-kelly-pct", style={
                        'fontSize': '48px', 'fontWeight': '900', 'color': '#d4af37'
                    }),
                    html.Div("Fractional 0.25x", style={
                        'fontSize': '10px', 'color': 'rgba(255,255,255,0.3)', 'marginTop': '8px'
                    })
                ], style={
                    'background': '#0a0a0a', 'border': '1px solid rgba(212, 175, 55, 0.2)',
                    'padding': '40px', 'textAlign': 'center'
                })
            ], width=3),
        ], style={'marginBottom': '48px'}),
        
        # ROW 2: Window Heatmap
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("24-HOUR WINDOW PROBABILITY MAP", style={
                        'fontSize': '10px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.3)',
                        'letterSpacing': '3px', 'marginBottom': '24px'
                    }),
                    dcc.Graph(id="window-heatmap", style={'height': '200px'})
                ], style={
                    'background': '#0a0a0a', 'border': '1px solid rgba(255,255,255,0.08)',
                    'padding': '32px'
                })
            ], width=12)
        ], style={'marginBottom': '48px'}),
        
        # ROW 3: Signal Breakdown
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("SIGNAL WEIGHTS", style={
                        'fontSize': '10px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.3)',
                        'letterSpacing': '3px', 'marginBottom': '24px'
                    }),
                    html.Div(id="window-signal-breakdown")
                ], style={
                    'background': '#0a0a0a', 'border': '1px solid rgba(255,255,255,0.08)',
                    'padding': '32px'
                })
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Div("REGIME TRANSITION HISTORY", style={
                        'fontSize': '10px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.3)',
                        'letterSpacing': '3px', 'marginBottom': '24px'
                    }),
                    html.Div(id="window-regime-history")
                ], style={
                    'background': '#0a0a0a', 'border': '1px solid rgba(255,255,255,0.08)',
                    'padding': '32px'
                })
            ], width=6)
        ])
    ], style={
        'padding': '48px',
        'background': 'linear-gradient(180deg, #000 0%, #0a0a0a 100%)',
        'minHeight': '100vh'
    })


def create_timing_page():
    return html.Div([
        html.Div([
            html.H1("Timing Infrastructure", style={'fontSize': '32px', 'fontWeight': '800', 'margin': 0}),
            html.Div("⏱️ Precision latency and behavioral extraction logs", style={'color': 'rgba(255,255,255,0.4)', 'fontSize': '14px', 'marginTop': '4px'})
        ], style={'marginBottom': '40px'}),
        
        html.Div([
            html.Div(id="timing-table-container")
        ], className="glass-card", style={'padding': '30px'})
    ])



app.layout = html.Div([
    dcc.Location(id='url'),
    dcc.Interval(id='refresh', interval=5000),  # 5s interval to reduce server load
    dcc.Store(id='language-store', data='en'),
    
    # Root Flex Container
    html.Div([
        # SIDEBAR (Fixed Left)
        create_sidebar(),

        # MAIN CONTENT AREA (Flex Grow)
        html.Div([
            # GLOBAL HEADER BAR (Pinned)
            html.Div([
                html.Div([
                    html.Span("REGIME STATUS: ", id="lbl-regime-status", style={'color': 'rgba(255,255,255,0.4)'}),
                    html.Span(id="hdr-regime", children="SEARCHING...", className="cyan")
                ], style={'flex': 1}),
                
                # SHA-256 Ticker
                html.Div([
                    html.Div(id="hdr-hash-stream", className="ticker", style={'fontSize': '10px', 'color': 'rgba(255,255,255,0.3)'})
                ], className="ticker-wrap", style={'flex': 2}),
                
                html.Div([
                    html.Span("SYS LATENCY: ", id="lbl-sys-latency", style={'color': 'rgba(255,255,255,0.4)'}),
                    html.Span(id="hdr-latency", children="12ms", className="cyan")
                ], style={'flex': 1, 'textAlign': 'right'}),
            ], style={
                'display': 'flex', 'alignItems': 'center', 'background': '#000', 
                'padding': '8px 16px', 'borderBottom': '1px solid var(--hud-border)'
            }),

            # DYNAMIC PAGE CONTENT
            html.Div(id='page-content', children=create_dashboard(), style={'flex': 1, 'display': 'flex', 'flexDirection': 'column', 'overflowY': 'auto'}),

            # GLOBAL FOOTER BAR (Pinned)
            html.Div([
                html.Div([
                    html.Span("SRC: ", style={'color': '#ffc107', 'fontWeight': 'bold'}),
                    html.Span(id="ftr-source-ip", children="...", className="cyan")
                ], style={'marginRight': '16px'}),
                html.Div([
                    html.Span("TARGET: ", id="lbl-target", style={'color': 'rgba(255,255,255,0.3)'}),
                    html.Span(id="ftr-target", children="2.0x", className="cyan")
                ], style={'marginRight': '16px'}),
                html.Div([
                    html.Span("CONSENSUS: ", id="lbl-consensus", style={'color': 'rgba(255,255,255,0.3)'}),
                    html.Span(id="ftr-consensus", children="75%", className="cyan")
                ], style={'marginRight': '16px'}),
                html.Div([
                    html.Span("SHA: ", style={'color': 'rgba(255,255,255,0.3)'}),
                    html.Span("OK", className="cyan")
                ]),
                html.Div(id="ftr-time", className="mono", style={'flex': 1, 'textAlign': 'right', 'fontSize': '10px', 'color': 'rgba(255,255,255,0.3)'})
            ], style={
                'display': 'flex', 'alignItems': 'center', 'background': '#000', 
                'padding': '8px 16px', 'borderTop': '1px solid var(--hud-border)', 'fontSize': '11px'
            })
        ], id="hud-main", className="hud-container", style={'flex': 1, 'display': 'flex', 'flexDirection': 'column'})
    ], style={'display': 'flex', 'height': '100vh', 'overflow': 'hidden'})
])



# ============ LOGIC HELPERS (Centralized for Master Update) ============

def update_whale_chart_logic():
    active_bets = betting_behavior.get('activeBets', [])
    buckets = {'SHRIMP (<1k)':0, 'FISH (1-5k)':0, 'DOLPHIN (5-20k)':0, 'SHARK (20-100k)':0, 'WHALE (>100k)':0}
    for b in active_bets:
        amt = b.get('amount', 0)
        if amt < 1000: buckets['SHRIMP (<1k)'] += amt
        elif amt < 5000: buckets['FISH (1-5k)'] += amt
        elif amt < 20000: buckets['DOLPHIN (5-20k)'] += amt
        elif amt < 100000: buckets['SHARK (20-100k)'] += amt
        else: buckets['WHALE (>100k)'] += amt
    fig = go.Figure(go.Scatterpolar(r=list(buckets.values())+[list(buckets.values())[0]], theta=list(buckets.keys())+[list(buckets.keys())[0]], fill='toself', line=dict(color='cyan')))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=40,r=40,t=20,b=20), height=300, polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=False)), showlegend=False)
    return fig

def update_chart_logic():
    data = crash_data[-40:]
    fig = go.Figure(go.Scatter(y=data, mode='lines+markers', line=dict(color='rgba(255,255,255,0.2)', shape='hv'), marker=dict(color=['cyan' if v>=2.0 else 'red' if v<1.2 else '#666' for v in data], size=4)))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=40,r=20,t=10,b=40), height=300, showlegend=False)
    return fig

def update_betting_window_logic():
    try:
        sig = get_signal()
        cons = sig.get('consensus_score', 0)
        v2_open = cons >= 0.75
        banner = html.Div("WINDOW OPEN" if v2_open else "ACCUMULATION", style={'fontSize':'72px', 'fontWeight':'900', 'color':'#d4af37' if v2_open else 'rgba(255,255,255,0.2)'})
        h_fig = go.Figure(go.Bar(y=[1]*24, marker_color=['#333']*24))
        h_fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=150, margin=dict(l=0,r=0,t=0,b=0))
        return banner, f"{cons*100:.0f}%", sig.get('regime','INIT'), f"{sig.get('win_probability',0.5)*100:.0f}%", f"{betting_behavior.get('poolSize',0)/50000:.2f}x", f"{min(5, cons*5):.1f}%", h_fig, [], []
    except: return html.Div("Calibrating..."), "0%", "INIT", "0%", "1.0x", "0%", go.Figure(), [], []

# ============ CALLBACKS ============

@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def route(path):
    if path == '/charts': return create_charts_page()
    if path == '/analysis': return create_analysis_page()
    if path == '/temporal': return create_temporal_page()
    if path == '/windows': return create_betting_window_page()
    if path == '/audit': return create_audit_page()
    if path == '/history': return create_history_page()
    if path == '/timing': return create_timing_page()
    return create_dashboard()



@app.callback(
    [
        # 1. Sidebar
        Output('sidebar-crashes', 'children'),
        Output('sidebar-crashes-val', 'children'),
        
        # 2. Header
        Output('hdr-regime', 'children'),
        Output('hdr-regime', 'className'),
        Output('hdr-hash-stream', 'children'),
        Output('hdr-latency', 'children'),
        
        # 3. Footer
        Output('ftr-target', 'children'),
        Output('ftr-consensus', 'children'),
        Output('ftr-time', 'children'),
        Output('ftr-source-ip', 'children'),
        
        # 4. HUD Primary
        Output('hud-main', 'className'),
        Output('signal-text', 'children'),
        Output('signal-info', 'children'),
        Output('consensus-pct', 'children'),
        Output('stat-target', 'children'),
        Output('metric-pof', 'children'),
        Output('metric-mv', 'children'),
        
        # 5. HUD Status
        Output('model-health-display', 'children'),
        Output('logic-scoreboard', 'children'),
        Output('narrative-display', 'children'),
        Output('stat-winrate', 'children'),
        Output('bet-balance', 'children'),
        Output('history-strip', 'children'),
        
        # 6. Charts & Probability
        Output('whale-chart', 'figure'),
        Output('main-chart', 'figure'),
        Output('pro-chart', 'figure'),
        Output('winrate-chart', 'figure'),
        Output('vol-chart', 'figure'),
        Output('dist-chart', 'figure'),
        Output('prob-matrix', 'figure'),
        
        # 7. Analysis & History
        Output('a-trend', 'children'),
        Output('a-pattern', 'children'),
        Output('a-regime', 'children'),
        Output('a-behavior', 'children'),
        Output('h-total', 'children'),
        Output('h-avg', 'children'),
        Output('h-max', 'children'),
        Output('h-min', 'children'),
        Output('history-grid', 'children'),
        
        # 8. V2 Window Page
        Output('window-status-banner', 'children'),
        Output('window-consensus-score', 'children'),
        Output('window-current-state', 'children'),
        Output('window-state-prob', 'children'),
        Output('window-pof-value', 'children'),
        Output('window-kelly-pct', 'children'),
        Output('window-heatmap', 'figure'),
        Output('window-signal-breakdown', 'children'),
        Output('window-regime-history', 'children'),
        
        # 9. Temporal Page
        Output('temporal-heatmap', 'figure'),
        Output('hourly-cycle-chart', 'figure'),
        Output('temporal-intel-content', 'children'),
        Output('temporal-alert-banner', 'children'),
        Output('best-hour-display', 'children'),
        Output('best-hour-avg', 'children'),
        Output('worst-hour-display', 'children'),
        Output('worst-hour-avg', 'children'),
        Output('instant-crash-rate', 'children'),
        Output('velocity-anomaly-count', 'children'),
        Output('high-stake-warning-container', 'children'),
        Output('velocity-anomaly-list', 'children'),
        
        # 10. Audit & Timing
        Output('audit-pl-chart', 'figure'),
        Output('audit-velocity-chart', 'figure'),
        Output('audit-stats-table', 'children'),
        Output('timing-table-container', 'children')
    ],
    [Input('refresh', 'n_intervals')],
    [State('language-store', 'data'), State('temporal-timeframe-selector', 'value')],
    prevent_initial_call=False
)
def update_master(n, lang, tf):
    """
    MASTER UPDATE CALLBACK - Consolidates 15+ callbacks into ONE request.
    This eliminates 502 Bad Gateway errors by preventing concurrent request 'stampedes'.
    """
    if n is None: return [dash.no_update] * 37
    
    try:
        # --- DATA PREP ---
        sig = get_signal()
        l = lang if lang in TRANSLATIONS else 'en'
        t = TRANSLATIONS[l]
        
        signal = sig.get('signal', 'WAIT')
        consensus = sig.get('consensus_score', 0) * 100
        analysis = sig.get('analysis', {})
        
        # Broadcast signal to WebSockets for zero-delay extension sync
        try:
            socketio.emit('signal_update', {
                'signal': signal,
                'target': sig.get('target', 2.0),
                'prob': sig.get('prob', 0),
                'bet': sig.get('bet', 0)
            })
        except: pass

        # --- 1. SIDEBAR ---
        crashes_count = str(len(crash_data))
        
        # --- 2. HEADER ---
        regime = sig.get('regime', 'NEUTRAL').upper()
        hdr_regime_class = "cyan" if "ACCUMULATION" not in regime else "red"
        
        round_id = betting_behavior.get('roundId', 'WAITING...')
        live_coef = betting_behavior.get('liveCoefficient', 0)
        last_coef = betting_behavior.get('lastCoefficient', 0)
        
        if live_coef > 0:
            game_state, coef_display = "🟢 LIVE", f"{live_coef:.2f}x"
        elif betting_behavior.get('totalBettors', 0) > 0:
            game_state, coef_display = "🟡 BETTING", "—"
        else:
            game_state, coef_display = "⚪ WAITING", f"Last: {last_coef:.2f}x" if last_coef else "—"
            
        hash_stream = f"{game_state}  •  Round: {round_id}  •  {coef_display}"
        latency = f"{10 + (n % 5)}ms"
        
        # --- 3. FOOTER ---
        target_display = f"{sig.get('target', 2.0)}x"
        consensus_pct_display = f"{consensus:.0f}%"
        ftr_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')
        ftr_ip = last_ip
        
        # --- 4. HUD PRIMARY ---
        pool_size = betting_behavior.get('poolSize', 0)
        pof_value = pool_size / 50000 if pool_size > 0 else 1.0
        
        hud_class = "hud-container"
        if consensus >= 75 or signal == 'BET': hud_class += " hud-flash-cyan"
        elif signal == 'SKIP': hud_class += " hud-flash-red"
        
        bet_amount = sig.get('bet', 0)
        info = f"TZS {bet_amount:,.0f}" if bet_amount > 0 else "TZS 0"
        
        pof_display = f"{pof_value:.1f}x"
        mv_display = "NORMAL" if pof_value < 1.5 else "FAST" if pof_value < 2.0 else "SPIKE"

        # NEW: Model Health Badges
        analysis = sig.get('analysis', {})
        health_badges = []
        # Core models to check
        core_models = ['trend', 'pattern', 'volatility', 'regime', 'statistical', 'behavioral', 'ml', 'temporal']
        for m in core_models:
            is_active = m in analysis and analysis[m] is not None
            status_color = '#10b981' if is_active else 'rgba(255,255,255,0.2)'
            health_badges.append(html.Div([
                html.Span("●", style={'color': status_color, 'fontSize': '8px', 'marginRight': '4px'}),
                html.Span(m.upper(), style={'fontSize': '9px', 'color': 'rgba(255,255,255,0.5)'})
            ], style={'background': 'rgba(255,255,255,0.05)', 'padding': '2px 6px', 'borderRadius': '4px'}))
        
        # 5. Terminal Log (Logic Scoreboard)
        scoreboard = []
        timestamp = datetime.now().strftime('%H:%M:%S')
        for name, res in analysis.items():
            if not res: continue
            res_color = 'var(--kinetic-cyan)' if res.get('signal') == 'BET' else 'var(--impact-red)' if res.get('signal') == 'SKIP' else 'rgba(255,255,255,0.3)'
            scoreboard.append(html.Div([
                html.Span(f"[{timestamp}] ", style={'color': 'rgba(255,255,255,0.2)'}),
                html.Span(f"{name.upper()}: ", style={'color': 'rgba(255,255,255,0.5)'}),
                html.Span(res.get('signal', 'WAIT'), style={'color': res_color, 'fontWeight': '700'})
            ], style={'padding': '2px 0', 'borderBottom': '1px solid rgba(255,255,255,0.02)'}))
        
        # 6. Session Stats
        winrate = f"{sig.get('prob', 0):.0f}%"
        b = betting_behavior
        balance = b.get('balance', 0)
        session_profit = f"TZS {balance:,.0f}" if balance > 0 else "TZS 0"
        
        # 7. History Strip - Show ALL crashes (scrollable)
        history_elements = []
        for i, val in enumerate(reversed(crash_data[-100:])):  # Show last 100 for performance
            is_latest = i == 0
            cls = "history-badge badge-cyan" if val >= 2.0 else "history-badge badge-red"
            if is_latest:
                # Special CSS class handles the highlight and thicker border
                history_elements.append(html.Div([
                    html.Span("NEW", style={'fontSize': '8px', 'fontWeight': '900', 'marginRight': '4px', 'opacity': '0.7'}),
                    html.Span(f"{val:.2f}x")
                ], className=f"{cls} badge-latest"))
            else:
                history_elements.append(html.Div(f"{val:.2f}x", className=cls))

        # 8. All Charts & Logic
        whale_fig = update_whale_chart_logic()
        main_fig = update_chart_logic()

        # --- Pro Chart Logic ---
        if len(crash_data) < 10:
            pro_fig = go.Figure()
            pro_fig.add_annotation(text="Need more data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color='rgba(255,255,255,0.3)'))
            pro_fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        else:
            data_np = np.array(crash_data)
            x = list(range(len(data_np)))
            ma10 = pd.Series(data_np).rolling(10).mean()
            ma30 = pd.Series(data_np).rolling(30).mean()
            std = pd.Series(data_np).rolling(20).std()
            
            pro_fig = go.Figure()
            pro_fig.add_trace(go.Scatter(x=x, y=ma10+2*std, mode='lines', line=dict(color='rgba(16,185,129,0.2)'), showlegend=False))
            pro_fig.add_trace(go.Scatter(x=x, y=ma10-2*std, mode='lines', line=dict(color='rgba(16,185,129,0.2)'), fill='tonexty', fillcolor='rgba(16,185,129,0.05)', showlegend=False))
            pro_fig.add_trace(go.Scatter(x=x, y=ma10, mode='lines', line=dict(color='#f59e0b', width=2), name='MA10'))
            pro_fig.add_trace(go.Scatter(x=x, y=ma30, mode='lines', line=dict(color='#10b981', width=2), name='MA30'))
            colors = ['#10b981' if c >= 2.0 else '#ef4444' for c in data_np]
            pro_fig.add_trace(go.Scatter(x=x, y=data_np, mode='markers+lines', marker=dict(color=colors, size=5), line=dict(color='rgba(255,255,255,0.2)', width=1), name='Crashes'))
            pro_fig.add_hline(y=2.0, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="2.0x")
            pro_fig.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=50, r=20, t=20, b=50), legend=dict(orientation='h', y=1.1),
                xaxis=dict(showgrid=False, title='Round'), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title='Multiplier'),
                hovermode='x unified'
            )

        # --- Sub Charts Logic (Winrate, Vol, Dist) ---
        empty_fig = go.Figure()
        empty_fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20,r=20,t=10,b=20))
        if len(crash_data) < 20:
            wr_fig, vol_fig, dist_fig = empty_fig, empty_fig, empty_fig
        else:
            data_np = np.array(crash_data)
            wins = (data_np >= 2.0).astype(int)
            
            # Win rate
            wr = pd.Series(wins).rolling(20).mean() * 100
            wr_fig = go.Figure()
            wr_fig.add_trace(go.Scatter(y=wr, mode='lines', fill='tozeroy', line=dict(color='#10b981'), fillcolor='rgba(16,185,129,0.2)'))
            wr_fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.2)")
            wr_fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                margin=dict(l=30,r=10,t=10,b=20), yaxis=dict(range=[0,100], showgrid=False), xaxis=dict(showgrid=False))
            
            # Vol
            vol = pd.Series(data_np).rolling(20).std()
            vol_fig = go.Figure()
            vol_fig.add_trace(go.Scatter(y=vol, mode='lines', fill='tozeroy', line=dict(color='#f59e0b'), fillcolor='rgba(245,158,11,0.2)'))
            vol_fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=30,r=10,t=10,b=20), yaxis=dict(showgrid=False), xaxis=dict(showgrid=False))
            
            # Dist
            dist_fig = go.Figure()
            dist_fig.add_trace(go.Histogram(x=data_np, nbinsx=20, marker_color='#3b82f6'))
            dist_fig.add_vline(x=2.0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            dist_fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=30,r=10,t=10,b=20), yaxis=dict(showgrid=False), xaxis=dict(showgrid=False))

        # --- Analysis Logic ---
        if not ADVANCED_PREDICTOR or len(crash_data) < 20:
            msg = html.Div("Need more data", style={'color': 'rgba(255,255,255,0.3)'})
            a_trend = a_pattern = a_regime = a_behavior = msg
        else:
            def fmt_analysis(a):
                if not a: return html.Div("N/A", style={'color': 'rgba(255,255,255,0.3)'})
                color = '#10b981' if a.get('signal')=='BET' else '#ef4444' if a.get('signal')=='SKIP' else '#f59e0b'
                return html.Div([
                    html.Div(a.get('signal', 'N/A'), style={'fontSize': '24px', 'fontWeight': '700', 'color': color}),
                    html.Div(a.get('reason', '').split('|')[0], style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '12px', 'marginTop': '8px'}),
                    html.Div(f"Confidence: {a.get('confidence', 0)*100:.0f}%", style={'color': 'rgba(255,255,255,0.3)', 'fontSize': '11px', 'marginTop': '4px'})
                ])
            
            a_trend = fmt_analysis(analysis.get('trend'))
            a_pattern = fmt_analysis(analysis.get('pattern'))
            a_regime = fmt_analysis(analysis.get('regime'))
            a_behavior = fmt_analysis(analysis.get('behavioral'))

        # --- Probability Matrix Logic ---
        targets = [1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
        probs = [np.mean(np.array(crash_data) >= t)*100 for t in targets] if crash_data else [0]*len(targets)
        colors = ['#10b981' if p > 50 else '#f59e0b' if p > 30 else '#ef4444' for p in probs]
        
        prob_fig = go.Figure()
        prob_fig.add_trace(go.Bar(x=[f'{t}x' for t in targets], y=probs, marker_color=colors, text=[f'{p:.0f}%' for p in probs], textposition='outside'))
        prob_fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                 margin=dict(l=40,r=20,t=20,b=40), yaxis=dict(range=[0,110], showgrid=False), xaxis=dict(showgrid=False))

        # --- History Stats Logic ---
        if not crash_data:
            h_total = "0"
            h_avg = "—"
            h_max = "—"
            h_min_val = "—"
            h_grid = []
        else:
            data_np = np.array(crash_data)
            h_total = str(len(crash_data))
            h_avg = f"{np.mean(data_np):.2f}x"
            h_max = f"{np.max(data_np):.2f}x"
            h_min_val = f"{np.min(data_np):.2f}x"
            
            h_grid = []
            for c in reversed(crash_data[-100:]):
                bg = 'linear-gradient(135deg, #10b981, #059669)' if c >= 3.0 else 'linear-gradient(135deg, #f59e0b, #d97706)' if c >= 2.0 else 'linear-gradient(135deg, #ef4444, #dc2626)'
                h_grid.append(html.Div(f"{c:.2f}x", style={
                    'background': bg, 'padding': '6px 12px', 'borderRadius': '8px', 'fontSize': '12px', 'fontWeight': '600'
                }))

        # 10. Temporal Logic
        # Fallback logic for secondary pages to reduce main thread load
        # In a real app we would only calculate this if the URL matches
        tf = tf or '24h'
        t_fig1 = t_fig2 = go.Figure()
        t_intel = t_alert = best_h = best_avg = worst_h = worst_avg = inst_r = anom_c = h_stake = anom_l = html.Div("...")

        # 11. Audit & Timing Logic
        pl_fig = vel_fig = go.Figure()
        audit_table = timing_table = html.Div("...")

        return (
            crashes_count, crashes_count, 
            regime, hdr_regime_class, hash_stream, latency,
            target_display, consensus_pct_display, ftr_time, ftr_ip,
            hud_class, signal, info, consensus_pct_display, target_display, pof_display, mv_display,
            health_badges, scoreboard, narrative, winrate, session_profit, history_elements,
            whale_fig, main_fig, pro_fig, wr_fig, vol_fig, dist_fig, prob_fig, 
            a_trend, a_pattern, a_regime, a_behavior, h_total, h_avg, h_max, h_min_val, h_grid,
            window_banner, v2_cons, v2_state, v2_prob, v2_pof_val, v2_kelly, h_fig, w_rows, h_rows,
            t_fig1, t_fig2, t_intel, t_alert, best_h, best_avg, worst_h, worst_avg, inst_r, anom_c, h_stake, anom_l,
            pl_fig, vel_fig, audit_table, timing_table
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [dash.no_update] * 64


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 ZEPPELIN PRO DASHBOARD (WebSocket Enabled)")
    print("="*50)
    print(f"\n✅ Advanced Predictor: {'Loaded' if ADVANCED_PREDICTOR else 'Not available'}")
    print(f"📊 Crashes: {len(crash_data)}")
    print(f"\n🌐 Open http://localhost:8050\n")
    socketio.run(server, debug=False, host='0.0.0.0', port=8050)
