"""
Zeppelin Dashboard Translations
Languages: English (en), Swahili (sw)
"""

TRANSLATIONS = {
    'en': {
        # Navigation
        'nav_dashboard': 'DASHBOARD',
        'nav_analytics': 'ANALYTICS',
        'nav_temporal': 'TEMPORAL',
        'nav_windows': 'WINDOWS',
        'nav_audit': 'AUDIT',
        'nav_history': 'HISTORY',
        'nav_timing': 'TIMING',

        # Header/Footer
        'hdr_regime': 'REGIME STATUS',
        'hdr_latency': 'SYS LATENCY',
        'ftr_target': 'TARGET',
        'ftr_consensus': 'CONSENSUS',
        
        # Modules
        'mod_command': 'COMMAND STATUS',
        'mod_consensus': 'CONSENSUS ENGINE',
        'mod_session': 'SESSION ANALYTICS',
        'mod_intelligence': 'INTELLIGENCE STRIP',
        'mod_perf': 'PERFORMANCE VECTOR',
        'mod_whale': 'WHALE RADAR',
        'mod_log': 'TERMINAL LOG STREAM',
        'mod_interpreter': 'STRATEGIC INTERPRETER',

        # Intelligence Strip
        'lbl_target_prob': 'TARGET PROBABILITY',
        'lbl_pool_load': 'POOL LOAD (POF)',
        'lbl_velocity': 'FLOW VELOCITY',
        'lbl_active_models': 'ACTIVE MODELS',
        'val_searching': 'SEARCHING...',

        # Whale Radar Labels
        'wr_shrimp': 'SHRIMP (<1k)',
        'wr_fish': 'FISH (1-5k)',
        'wr_dolphin': 'DOLPHIN (5-20k)',
        'wr_shark': 'SHARK (20-100k)',
        'wr_whale': 'WHALE (>100k)',

        # Interpretations (Narrative)
        'narrative_wait': "System is analyzing market conditions. Please wait for clear patterns.",
        'narrative_bet': "Market conditions are favorable. Probability of reaching 2.0x is high.",
        'narrative_skip': "Market is effectively dangerous. High risk of early crash detected.",
        'narrative_danger': "⚠️ DANGER: Whale activity or instability detected. Do not bet.",
        
        # Tooltips (Tour)
        'tip_command': 'Shows the final decision: BET, SKIP, or WAIT. Follow this for simple play.',
        'tip_consensus': 'Percentage of models agreeing on the current prediction.',
        'tip_whale': 'Visualizes real-time money flow. Large spikes mean "Whales" (rich players) are betting.',
        'tip_perf': 'Real-time graph of crash points. Dashed line is the 2.0x target.',
        'tip_log': 'Live feed of the internal decision logic from all AI models.',
        'tip_pof': 'Pool Overload Factor. High values (>1.5x) mean the pot is too big and likely to crash early.',
        
        # Session Analytics
        'lbl_winrate': 'WIN RATE',
        'lbl_profit': 'SESSION PROFIT',
        'lbl_total_logs': 'TOTAL LOGS',
        'lbl_optimal': 'OPTIMAL TARGET: ',
        'lbl_edge': 'EDGE ESTIMATE',
        'lbl_load_alloc': 'LOAD ALLOC: ',
        
        # Header/Footer Labels
        'lbl_regime_status': 'REGIME STATUS: ',
        'lbl_sys_latency': 'SYS LATENCY: ',
        'lbl_target': 'TARGET: ',
        'lbl_consensus': 'CONSENSUS: ',
        
        # Signal States
        'sig_bet': 'BET',
        'sig_skip': 'SKIP',
        'sig_wait': 'WAIT',
        'sig_searching': 'SEARCHING...',
        
        # Charts Page
        'page_charts_title': 'Charts',
        'lbl_crash_mult': '📈 Crash Multiplier',
        'lbl_chart_desc': ' • With MA10, MA30 & Bollinger Bands',
        'lbl_winrate_trend': 'Win Rate Trend',
        'lbl_volatility': 'Volatility',
        'lbl_distribution': 'Distribution',
        
        # Audit Page
        'page_audit_title': 'Audit Hub',
        'page_audit_desc': '🛡️ Real-time house integrity and payout auditing',
        'lbl_house_pl': 'House P&L (Recent Rounds)',
        'lbl_velocity_audit': 'Velocity Audit (ms/step)',
        'lbl_deep_audit': '🔍 Deep Audit Table',
        'lbl_audit_desc': ' • Last 20 Rounds with Seed Verification',
        
        # Temporal Page
        'page_temporal_title': '🕐 Time Myths & Temporal Intelligence',
        'page_temporal_desc': 'Real-time insights from temporal pattern analysis',
        'lbl_timeframe': 'Analysis Timeframe:',
    },
    'sw': {
        # Navigation
        'nav_dashboard': 'DASHBODI',
        'nav_analytics': 'UCHAMBUZI',
        'nav_temporal': 'MUDA',
        'nav_windows': 'MADIRISHA',
        'nav_audit': 'UKAGUZI',
        'nav_history': 'HISTORIA',
        'nav_timing': 'MUDA HALISI',

        # Header/Footer
        'hdr_regime': 'HALI YA MCHEZO',
        'hdr_latency': 'KASI YA MFUMO',
        'ftr_target': 'LENGO',
        'ftr_consensus': 'MAKUBALIANO',

        # Modules
        'mod_command': 'AMRI KUU',
        'mod_consensus': 'MASHINE YA UAMUZI',
        'mod_session': 'TAKWIMU ZA KIPINDI',
        'mod_intelligence': 'RIBONI YA AKILI',
        'mod_perf': 'MWELEKEO WA MCHEZO',
        'mod_whale': 'RADA YA "WANGWE"',
        'mod_log': 'MANDO YA MATAARIFA',
        'mod_interpreter': 'MCHAMBUZI MKUU',

        # Intelligence Strip
        'lbl_target_prob': 'UWEZEKANO WA LENGO',
        'lbl_pool_load': 'MZIGO WA PESA (POF)',
        'lbl_velocity': 'KASI YA PESA',
        'lbl_active_models': 'MODELI HAI',
        'val_searching': 'INATAFUTA...',

        # Whale Radar Labels
        'wr_shrimp': 'UDUVU (<1k)',
        'wr_fish': 'SAMAKI (1-5k)',
        'wr_dolphin': 'POMBOO (5-20k)',
        'wr_shark': 'PAPA (20-100k)',
        'wr_whale': 'WANGWE (>100k)',

        # Interpretations (Narrative)
        'narrative_wait': "Mfumo unachunguza soko. Subiri mpaka uone dalili nzuri.",
        'narrative_bet': "Hali ya soko ni nzuri. Nafasi ya kufika 2.0x ni kubwa.",
        'narrative_skip': "Soko ni la hatari sasa. Kuna uwezekano mkubwa wa kuungua mapema.",
        'narrative_danger': "⚠️ HATARI: 'Wangwe' wameingia au soko halijatulia. Usibeti.",

        # Tooltips (Tour)
        'tip_command': 'Hapa ndipo uamuzi wa mwisho: BET (Weka), SKIP (Ruka), au WAIT (Subiri).',
        'tip_consensus': 'Asilimia ya modeli zinazokubaliana na uamuzi huu.',
        'tip_whale': 'Inaonyesha mtiririko wa pesa. Michoro mikubwa inamaanisha "Wangwe" (matajiri) wanaweka pesa.',
        'tip_perf': 'Grafu ya mchezo. Mstari wa nukta-nukta ni lengo la 2.0x.',
        'tip_log': 'Taarifa za moja kwa moja kuhusu jinsi kompyuta inavyofanya maamuzi.',
        'tip_pof': 'Kipimo cha Mzigo. Namba kubwa (>1.5x) inamaanisha pesa ni nyingi mezani na inaweza kuungua haraka.',
        
        # Session Analytics
        'lbl_winrate': 'KIWANGO CHA USHINDI',
        'lbl_profit': 'FAIDA YA KIPINDI',
        'lbl_total_logs': 'JUMLA YA KUMBUKUMBU',
        'lbl_optimal': 'LENGO BORA: ',
        'lbl_edge': 'MAKADIRIO YA FAIDA',
        'lbl_load_alloc': 'PESA ILIYOWEKWA: ',
        
        # Header/Footer Labels
        'lbl_regime_status': 'HALI YA MCHEZO: ',
        'lbl_sys_latency': 'KASI YA MFUMO: ',
        'lbl_target': 'LENGO: ',
        'lbl_consensus': 'MAKUBALIANO: ',
        
        # Signal States
        'sig_bet': 'WEKA',
        'sig_skip': 'RUKA',
        'sig_wait': 'SUBIRI',
        'sig_searching': 'INATAFUTA...',
        
        # Charts Page
        'page_charts_title': 'Grafu',
        'lbl_crash_mult': '📈 Kiwango cha Mchezo',
        'lbl_chart_desc': ' • Na MA10, MA30 na Bollinger Bands',
        'lbl_winrate_trend': 'Mwenendo wa Ushindi',
        'lbl_volatility': 'Msukosuko',
        'lbl_distribution': 'Usambazaji',
        
        # Audit Page
        'page_audit_title': 'Kituo cha Ukaguzi',
        'page_audit_desc': '🛡️ Ukaguzi wa uhalali wa nyumba na malipo',
        'lbl_house_pl': 'Faida/Hasara ya Nyumba (Raundi za Hivi Karibuni)',
        'lbl_velocity_audit': 'Ukaguzi wa Kasi (ms/hatua)',
        'lbl_deep_audit': '🔍 Jedwali la Ukaguzi wa Kina',
        'lbl_audit_desc': ' • Raundi 20 za Mwisho na Uthibitisho wa Mbegu',
        
        # Temporal Page
        'page_temporal_title': '🕐 Hadithi za Muda na Akili ya Muda',
        'page_temporal_desc': 'Maarifa ya wakati halisi kutoka kwa uchambuzi wa mifumo ya muda',
        'lbl_timeframe': 'Kipindi cha Uchambuzi:',
    }
}
