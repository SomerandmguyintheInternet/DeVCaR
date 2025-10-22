'''from __future__ import annotations
import sys
import os
import json
import re
import traceback
from contextlib import contextmanager
import uuid
import sqlite3
import secrets
import hashlib
import yaml
import threading
from queue import Queue
from datetime import datetime, timedelta
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, NamedTuple, Union, Set, Callable
from cryptography.fernet import Fernet
from pymongo import MongoClient
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, JSON
from sqlalchemy.orm import Session, declarative_base
import requests
try:
    from gql import Client, gql  # type: ignore
    from gql.transport.requests import RequestsHTTPTransport  # type: ignore
    _HAS_GQL = True
except Exception:
    Client = None
    gql = None
    RequestsHTTPTransport = None
    _HAS_GQL = False
from oauthlib.oauth2 import BackendApplicationClient
try:
    from requests_oauthlib import OAuth2Session
except Exception:
    OAuth2Session = None

# Default extension set for dev files
DEV_EXTENSIONS = ['.dev', '.devc', '.devkpi', '.devsdk']
 
# Guarded Qt imports: prefer PyQt6, then PySide6. Provide light dummies for static analysis
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
        QTextEdit, QListWidget, QTabWidget, QLineEdit, QPushButton, QStatusBar,
        QMenuBar, QAction, QToolBar, QMessageBox, QFileDialog, QInputDialog, QMenu,
        QSplitter, QTreeWidget, QTreeWidgetItem, QGroupBox, QFormLayout, QLabel,
        QTableWidget, QHeaderView, QComboBox, QCheckBox, QDialog, QScrollArea, QToolBox
    )
    from PyQt6.QtGui import QFont, QIcon, QPalette, QColor, QFileSystemModel
    from PyQt6.QtCore import Qt, QProcess, QUrl
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    _QT_AVAILABLE = True
except ImportError:
    try:
        from PySide6.QtWidgets import (  # type: ignore
            QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
            QTextEdit, QListWidget, QTabWidget, QLineEdit, QPushButton, QStatusBar,
            QMenuBar, QAction, QToolBar, QMessageBox, QFileDialog, QInputDialog, QMenu,
            QSplitter, QTreeWidget, QTreeWidgetItem, QGroupBox, QFormLayout, QLabel,
            QTableWidget, QHeaderView, QComboBox, QCheckBox, QDialog, QScrollArea, QToolBox
        )
        from PySide6.QtGui import QFont, QIcon, QPalette, QColor, QFileSystemModel  # type: ignore
        from PySide6.QtCore import Qt, QProcess, QUrl  # type: ignore
        from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore
        _QT_AVAILABLE = True
    except ImportError:
        _QT_AVAILABLE = False
        # Define dummy classes for type hinting and static analysis
        class QApplication: pass
        class QMainWindow: pass
        class QWidget: pass
        class QHBoxLayout: pass
        class QVBoxLayout: pass
        class QGridLayout: pass
        class QTextEdit: pass
        class QListWidget: pass
        class QTabWidget: pass
        class QLineEdit: pass
        class QPushButton: pass
        class QStatusBar: pass
        class QMenuBar: pass
        class QAction: pass
        class QToolBar: pass
        class QMessageBox: pass
        class QFileDialog: pass
        class QInputDialog: pass
        class QMenu: pass
        class QSplitter: pass
        class QTreeWidget: pass
        class QTreeWidgetItem: pass
        class QGroupBox: pass
        class QFormLayout: pass
        class QLabel: pass
        class QTableWidget: pass
        class QHeaderView: pass
        class QComboBox: pass
        class QCheckBox: pass
        class QFont: pass
        class QIcon: pass
        class QPalette: pass
        class QColor: pass
        class QFileSystemModel: pass
        class Qt: pass
        class QProcess: pass
        class QUrl: pass
        class QWebEngineView: pass
        class QDialog: pass
        class QScrollArea: pass
        class QToolBox: pass

# --- Fallback: Use devcar_renderer.py if Qt is not available ---
if not _QT_AVAILABLE:
    print("[INFO] PyQt6/PySide6 not available. Falling back to HTML rendering using devcar_renderer.py.")
    try:
        import os
        import sys
        import importlib.util
        renderer_path = os.path.join(os.path.dirname(__file__), 'devcar_renderer.py')
        spec = importlib.util.spec_from_file_location('devcar_renderer', renderer_path)
        devcar_renderer = importlib.util.module_from_spec(spec)
        sys.modules['devcar_renderer'] = devcar_renderer
        spec.loader.exec_module(devcar_renderer)
    except Exception as e:
        print(f"[ERROR] Could not import devcar_renderer.py: {e}")
        devcar_renderer = None

    def render_devcar_html(source: str):
        if devcar_renderer is not None:
            html = devcar_renderer.render_string(source)
            out_path = os.path.join(os.getcwd(), 'devcar_rendered.html')
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"[INFO] Rendered HTML written to {out_path}")
        else:
            print("[ERROR] devcar_renderer is not available. Cannot render HTML.")

    # Example usage: render_devcar_html("TITLE[\"Hello DevCAR\"]\nQUICKPICK[\"Run\", primary]")
    # You can call render_devcar_html from your main logic if Qt is not available.
class MetadataStore:
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        os.makedirs(os.path.join(self.workspace_path, 'applications'), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_path, 'pages'), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_path, 'regions'), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_path, 'items'), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_path, 'dynamic_actions'), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_path, 'processes'), exist_ok=True)

        self.db_path = os.path.join(self.workspace_path, 'studio.db')
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        c = self.conn.cursor()
        # Create minimal schema used by the IDE
        c.execute('''CREATE TABLE IF NOT EXISTS applications (id TEXT PRIMARY KEY, name TEXT, mode TEXT, theme TEXT, created_at TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS pages (id TEXT PRIMARY KEY, app_id TEXT, name TEXT, mode TEXT, metadata TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS regions (id TEXT PRIMARY KEY, page_id TEXT, type TEXT, title TEXT, source TEXT, metadata TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, region_id TEXT, name TEXT, type TEXT, required INTEGER, metadata TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS dynamic_actions (id TEXT PRIMARY KEY, page_id TEXT, name TEXT, when_event TEXT, where_target TEXT, what_actions TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS processes (id TEXT PRIMARY KEY, page_id TEXT, name TEXT, point TEXT, source TEXT, condition TEXT, timeout INTEGER)''')
        c.execute('''CREATE TABLE IF NOT EXISTS errors (id TEXT PRIMARY KEY, timestamp TEXT, app_id TEXT, page_id TEXT, region_id TEXT, action_id TEXT, sql_hash TEXT, error_type TEXT, message TEXT, stack_trace TEXT)''')
        self.conn.commit()

    def create_application(self, name: str, mode: str) -> str:
        """Create a new application"""
        app_id = str(uuid.uuid4())
        c = self.conn.cursor()
        c.execute(
            'INSERT INTO applications (id, name, mode) VALUES (?, ?, ?)',
            (app_id, name, mode)
        )
        self.conn.commit()
        
        # Create application metadata file
        metadata = {
            'id': app_id,
            'name': name,
            'mode': mode,
            'theme': 'default',
            'pages': [],
            'processes': {
                'startup': [],
                'shutdown': []
            }
        }
        with open(os.path.join(self.workspace_path, 'applications', f'{app_id}.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        return app_id

    def create_page(self, app_id: str, name: str) -> str:
        """Create a new page in an application"""
        page_id = str(uuid.uuid4())
        c = self.conn.cursor()
        c.execute(
            'INSERT INTO pages (id, app_id, name) VALUES (?, ?, ?)',
            (page_id, app_id, name)
        )
        self.conn.commit()
        
        # Create page metadata file
        metadata = {
            'id': page_id,
            'name': name,
            'regions': [],
            'dynamic_actions': [],
            'processes': {
                'before_header': [],
                'after_header': [],
                'before_submit': [],
                'after_submit': []
            }
        }
        with open(os.path.join(self.workspace_path, 'pages', f'{page_id}.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        return page_id

    # --- Metadata accessors ---
    def get_applications(self) -> List[dict]:
        c = self.conn.cursor()
        c.execute('SELECT id, name, mode, theme, created_at FROM applications')
        rows = c.fetchall()
        apps = []
        for r in rows:
            apps.append({'id': r[0], 'name': r[1], 'mode': r[2], 'theme': r[3], 'created_at': r[4]})
        return apps

    def get_application(self, app_id: str) -> dict:
        # Try JSON metadata first
        json_path = os.path.join(self.workspace_path, 'applications', f'{app_id}.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

        c = self.conn.cursor()
        c.execute('SELECT id, name, mode, theme FROM applications WHERE id = ?', (app_id,))
        row = c.fetchone()
        if not row:
            raise KeyError(f'Application not found: {app_id}')
        return {'id': row[0], 'name': row[1], 'mode': row[2], 'theme': row[3]}

    def get_pages(self, app_id: str) -> List[dict]:
        c = self.conn.cursor()
        c.execute('SELECT id, app_id, name, mode, metadata FROM pages WHERE app_id = ?', (app_id,))
        rows = c.fetchall()
        pages = []
        for r in rows:
            page = {'id': r[0], 'app_id': r[1], 'name': r[2], 'mode': r[3]}
            # try load page json metadata file
            json_path = os.path.join(self.workspace_path, 'pages', f"{r[0]}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        page_meta = json.load(f)
                        page.update(page_meta)
                except Exception:
                    pass
            pages.append(page)
        return pages

    def get_page(self, page_id: str) -> dict:
        json_path = os.path.join(self.workspace_path, 'pages', f'{page_id}.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        c = self.conn.cursor()
        c.execute('SELECT id, app_id, name, mode, metadata FROM pages WHERE id = ?', (page_id,))
        row = c.fetchone()
        if not row:
            raise KeyError(f'Page not found: {page_id}')
        return {'id': row[0], 'app_id': row[1], 'name': row[2], 'mode': row[3], 'metadata': row[4]}

    def get_regions(self, page_id: str) -> List[dict]:
        c = self.conn.cursor()
        c.execute('SELECT id, page_id, type, title, source, metadata FROM regions WHERE page_id = ?', (page_id,))
        rows = c.fetchall()
        regions = []
        for r in rows:
            reg = {'id': r[0], 'page_id': r[1], 'type': r[2], 'title': r[3], 'source': r[4]}
            try:
                reg_meta = json.loads(r[5]) if r[5] else {}
                reg.update({'metadata': reg_meta})
            except Exception:
                reg.update({'metadata': {}})
            regions.append(reg)
        return regions

    def get_items(self, region_id: str) -> List[dict]:
        c = self.conn.cursor()
        c.execute('SELECT id, region_id, name, type, required, metadata FROM items WHERE region_id = ?', (region_id,))
        rows = c.fetchall()
        items = []
        for r in rows:
            it = {'id': r[0], 'region_id': r[1], 'name': r[2], 'type': r[3], 'required': bool(r[4])}
            try:
                it_meta = json.loads(r[5]) if r[5] else {}
                it.update({'metadata': it_meta})
            except Exception:
                it.update({'metadata': {}})
            items.append(it)
        return items

    def get_dynamic_actions(self, page_id: str) -> List[dict]:
        c = self.conn.cursor()
        c.execute('SELECT id, page_id, name, when_event, where_target, what_actions FROM dynamic_actions WHERE page_id = ?', (page_id,))
        rows = c.fetchall()
        actions = []
        for r in rows:
            try:
                actions_list = json.loads(r[5]) if r[5] else []
            except Exception:
                actions_list = []
            actions.append({'id': r[0], 'page_id': r[1], 'name': r[2], 'when_event': r[3], 'where_target': r[4], 'what_actions': actions_list})
        return actions

    def get_processes(self, page_id: str) -> List[dict]:
        c = self.conn.cursor()
        c.execute('SELECT id, page_id, name, point, source, condition, timeout FROM processes WHERE page_id = ?', (page_id,))
        rows = c.fetchall()
        procs = []
        for r in rows:
            procs.append({'id': r[0], 'page_id': r[1], 'name': r[2], 'point': r[3], 'source': r[4], 'condition': r[5], 'timeout': r[6]})
        return procs

    def update_page(self, page_id: str, components: List[dict]):
        """Update page metadata JSON and DB metadata if present"""
        json_path = os.path.join(self.workspace_path, 'pages', f'{page_id}.json')
        page = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    page = json.load(f)
            except Exception:
                page = {}

        page['id'] = page_id
        page['components'] = components
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(page, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


class RuntimeRenderer:
    """Renders UI components based on metadata"""
    def __init__(self, mode: str):
        self.mode = mode
        self._load_theme()
        
    def _load_theme(self):
        """Load theme tokens"""
        self.theme = {
            'primary': '#3b82f6',
            'surface': '#ffffff',
            'text': '#1f2937',
            'danger': '#ef4444',
            'success': '#10b981'
        }
        
    def render_page(self, page_metadata: dict) -> Union[str, Any]:
        """Generate HTML or Qt widgets based on mode"""
        if self.mode == 'browser':
            return self._render_browser_page(page_metadata)
        else:
            return self._render_windows_page(page_metadata)
            
    def _render_browser_page(self, metadata: dict) -> str:
        """Generate HTML/Tailwind markup"""
        html = []
        for region in metadata['regions']:
            html.append(self._render_browser_region(region))
            
        return self._wrap_html(
            '\n'.join(html),
            styles=self._theme_to_tailwind(),
            scripts=self._get_required_scripts(metadata)
        )
        
    def _render_windows_page(self, metadata: dict) -> Any:
        """Generate Qt widget hierarchy"""
        container = QWidget()
        layout = QVBoxLayout()
        
        for region in metadata['regions']:
            widget = self._render_windows_region(region)
            layout.addWidget(widget)
            
        container.setLayout(layout)
        self._apply_theme_to_widget(container)
        return container
        
    def _theme_to_tailwind(self) -> dict:
        """Convert theme tokens to Tailwind classes"""
        return {
            'primary': self.theme['primary'],
            'surface': self.theme['surface'],
            'text': self.theme['text'],
            'danger': self.theme['danger'],
            'success': self.theme['success']
        }
        
    def _apply_theme_to_widget(self, widget: Any):
        """Apply theme to Qt widget"""
        palette = widget.palette()
        palette.setColor(QPalette.Window, QColor(self.theme['surface']))
        palette.setColor(QPalette.WindowText, QColor(self.theme['text']))
        palette.setColor(QPalette.Button, QColor(self.theme['primary']))
        palette.setColor(QPalette.ButtonText, QColor('#ffffff'))
        widget.setPalette(palette)

class ActionEngine:
    """Executes WHEN/WHERE/WHAT chains"""
    def __init__(self):
        self.transaction_manager = TransactionManager()
        
    def execute(self, action_chain: List[dict], context: dict):
        """Execute action chain with transaction support"""
        with self.transaction_manager.transaction():
            for action in action_chain:
                try:
                    result = self._execute_single_action(action, context)
                    if action.get('stop_on_error', True) and not result['success']:
                        self.transaction_manager.rollback()
                        return result
                except Exception as e:
                    if action.get('stop_on_error', True):
                        self.transaction_manager.rollback()
                        raise
                    context['errors'].append(str(e))
            self.transaction_manager.commit()
            
    def _execute_single_action(self, action: dict, context: dict) -> dict:
        """Execute a single action in the chain"""
        action_type = action['type']
        
        if action_type == 'validate':
            return self._validate_items(action['items'], context)
            
        elif action_type == 'sql':
            return self._execute_sql(
                action['connector'],
                action['text'],
                self._resolve_binds(action['binds'], context)
            )
            
        elif action_type == 'refresh':
            return self._refresh_region(action['target'])
            
        elif action_type == 'notify':
            return self._show_notification(
                action['text'],
                action.get('level', 'info')
            )
            
        return {'success': False, 'error': f'Unknown action type: {action_type}'}
        
    def _validate_items(self, items: List[str], context: dict) -> dict:
        """Validate form items"""
        errors = []
        for item_name in items:
            value = context['items'].get(item_name)
            if value is None or value == '':
                errors.append(f'{item_name} is required')
                
        return {
            'success': len(errors) == 0,
            'errors': errors
        }
        
    def _execute_sql(self, connector: str, sql: str, binds: dict) -> dict:
        """Execute SQL with bind variables"""
        try:
            with self.transaction_manager.get_connection(connector) as conn:
                cursor = conn.cursor()
                cursor.execute(sql, binds)
                return {'success': True, 'rows': cursor.rowcount}
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def _resolve_binds(self, bind_names: List[str], context: dict) -> dict:
        """Resolve bind variable values from context"""
        return {name: context['items'].get(name) for name in bind_names}

class TransactionManager:
    """Manages database transactions across connectors"""
    def __init__(self):
        self.connections = {}
        
    @contextmanager
    def transaction(self):
        """Transaction context manager"""
        try:
            yield self
        except Exception:
            self.rollback()
            raise
            
    def get_connection(self, connector: str) -> Any:
        """Get connection for a connector (create if needed)"""
        if connector not in self.connections:
            self.connections[connector] = self._create_connection(connector)
        return self.connections[connector]
        
    def commit(self):
        """Commit all active transactions"""
        for conn in self.connections.values():
            conn.commit()
            
    def rollback(self):
        """Rollback all active transactions"""
        for conn in self.connections.values():
            conn.rollback()
            
    def _create_connection(self, connector: str) -> Any:
        """Create new database connection"""
        # Load connector config and create appropriate connection
        config = self._load_connector_config(connector)
        if config['type'] == 'mysql':
            return self._create_mysql_connection(config)
        elif config['type'] == 'postgres':
            return self._create_postgres_connection(config)
        elif config['type'] == 'oracle':
            return self._create_oracle_connection(config)
        elif config['type'] == 'mongodb':
            return self._create_mongodb_connection(config)
        raise ValueError(f'Unknown connector type: {config["type"]}')
        
    def _load_connector_config(self, connector: str) -> dict:
        """Load connector configuration"""
        # Load from datasources/connectors.json
        if not os.path.exists('datasources/connectors.json'):
            raise FileNotFoundError('datasources/connectors.json not found')
        with open('datasources/connectors.json', 'r', encoding='utf-8') as f:
            connectors = json.load(f)
        return connectors[connector]

class ConnectorManager:
    """Manages database and API connections"""
    def __init__(self):
        self._load_connectors()
        self._init_connection_pools()
        
    def _load_connectors(self):
        """Load connector configurations"""
        if not os.path.exists('datasources/connectors.json'):
            self.connectors = {}
            return
        with open('datasources/connectors.json', 'r', encoding='utf-8') as f:
            self.connectors = json.load(f)
            
    def _init_connection_pools(self):
        """Initialize connection pools for each connector"""
        self.pools = {}
        for name, config in self.connectors.items():
            if config['type'] in ('mysql', 'postgres', 'oracle'):
                self.pools[name] = self._create_sql_pool(config)
            elif config['type'] == 'mongodb':
                self.pools[name] = self._create_mongo_pool(config)
                
    def _create_sql_pool(self, config: dict):
        """Create SQL connection pool"""
        url = config['url'].format(
            user=os.getenv(config['username']),
            password=os.getenv(config['password'])
        )
        return create_engine(url, pool_size=config.get('pool', {}).get('max', 5))
        
    def _create_mongo_pool(self, config: dict):
        """Create MongoDB connection pool"""
        client = MongoClient(
            config['uri'].format(
                user=os.getenv(config['username']),
                password=os.getenv(config['password'])
            )
        )
        return client[config['database']]
        
    def execute_query(self, connector: str, query: str, binds: dict = None) -> List[dict]:
        """Execute SQL query with bind variables"""
        pool = self.pools[connector]
        with Session(pool) as session:
            result = session.execute(query, binds or {})
            return [dict(row) for row in result]
            
    def execute_api_call(self, connector: str, endpoint: str, params: dict = None) -> dict:
        """Execute REST/GraphQL API call"""
        config = self.connectors[connector]
        if config['type'] == 'rest':
            return self._execute_rest_call(config, endpoint, params)
        elif config['type'] == 'graphql':
            return self._execute_graphql_call(config, endpoint, params)
            
    def _execute_rest_call(self, config: dict, endpoint: str, params: dict = None) -> dict:
        """Execute REST API call with authentication"""
        auth = config.get('auth', {})
        
        if auth['type'] == 'oauth2':
            client = BackendApplicationClient(client_id=os.getenv(auth['client_id']))
            oauth = OAuth2Session(client=client)
            token = oauth.fetch_token(
                token_url=auth['token_url'],
                client_id=os.getenv(auth['client_id']),
                client_secret=os.getenv(auth['client_secret'])
            )
            
        url = f"{config['base_url']}{endpoint}"
        response = requests.get(url, params=params, headers={'Authorization': f'Bearer {token["access_token"]}'})
        return response.json()
        
    def _execute_graphql_call(self, config: dict, query: str, variables: dict = None) -> dict:
        """Execute GraphQL query"""
        # Build headers
        headers = {}
        token_env = config.get('token')
        if token_env and os.getenv(token_env):
            headers['Authorization'] = f'Bearer {os.getenv(token_env)}'

        # If gql library is available, use it for better transport handling
        if _HAS_GQL and Client is not None and RequestsHTTPTransport is not None:
            transport = RequestsHTTPTransport(
                url=config['url'],
                headers=headers,
                verify=True,
                retries=3,
            )
            client = Client(transport=transport, fetch_schema_from_transport=True)
            result = client.execute(query if isinstance(query, str) else str(query), variable_values=variables)
            return result

        # Fallback: use requests to POST to the GraphQL endpoint
        payload = {
            'query': query if isinstance(query, str) else str(query),
            'variables': variables or {}
        }
        resp = requests.post(config['url'], json=payload, headers=headers)
        try:
            resp.raise_for_status()
        except Exception as e:
            return {'error': str(e), 'status_code': resp.status_code, 'text': resp.text}
        try:
            return resp.json()
        except ValueError:
            return {'text': resp.text}

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, List, Optional, Union
import re

# --- Lexer ---
class TokenType(Enum):
    # Control flow
    IF = auto()
    ELSE = auto()
    ENDIF = auto()
    FOR = auto()
    EACH = auto()
    IN = auto()
    ENDFOR = auto()
    
    # Components
    COMPONENT = auto()
    END_COMPONENT = auto()
    FORM = auto()
    END_FORM = auto()
    
    # UI Elements
    TITLE = auto()
    INPUT = auto()
    CHECKBOX = auto()
    QUICKPICK = auto()
    ICON = auto()
    ITEM_LIST = auto()
    CREATE_TABLE = auto()
    NAVBAR = auto()
    CHART = auto()
    
    # Data
    SAVE_DATA = auto()
    LOAD_DATA = auto()
    DEFINE_VAR = auto()
    
    # Operators and literals
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    DOT = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    EQUALS = auto()
    COMMA = auto()
    
    # Comparison operators
    EQEQ = auto()
    BANGEQ = auto()
    LT = auto()
    LTEQ = auto()
    GT = auto()
    GTEQ = auto()
    BANG = auto()
    
    EOF = auto()

@dataclass
class Token:
    type: TokenType
    lexeme: str
    literal: Any
    line: int
    column: int

class DevCarLexer:
    def __init__(self, source: str):
        self.source = source
        self.tokens: List[Token] = []
        self.start = 0
        self.current = 0
        self.line = 1
        self.column = 1
        
    def lex(self) -> List[Token]:
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        
        self.tokens.append(Token(TokenType.EOF, "", None, self.line, self.column))
        return self.tokens
    
    def scan_token(self):
        c = self.advance()
        
        # Handle block comments
        if c == '#' and self.peek() == '#':
            self.advance()  # consume second '#'
            while not self.is_at_end():
                if self.peek() == '#' and self.peek_next() == '#':
                    self.advance()  # consume first closing #
                    self.advance()  # consume second closing #
                    return
                if self.peek() == '\n':
                    self.line += 1
                    self.column = 1
                self.advance()
            return
            
        if c.isspace():
            if c == '\n':
                self.line += 1
                self.column = 1
            return
            
        if c.isalpha():
            self.identifier()
            return
            
        if c.isdigit():
            self.number()
            return
            
        match c:
            case '[':
                self.add_token(TokenType.LBRACKET)
            case ']':
                self.add_token(TokenType.RBRACKET)
            case '.':
                self.add_token(TokenType.DOT)
            case ',':
                self.add_token(TokenType.COMMA)
            case '=':
                if self.match('='):
                    self.add_token(TokenType.EQEQ)
                else:
                    self.add_token(TokenType.EQUALS)
            case '!':
                if self.match('='):
                    self.add_token(TokenType.BANGEQ)
                else:
                    self.add_token(TokenType.BANG)
            case '<':
                if self.match('='):
                    self.add_token(TokenType.LTEQ)
                else:
                    self.add_token(TokenType.LT)
            case '>':
                if self.match('='):
                    self.add_token(TokenType.GTEQ)
                else:
                    self.add_token(TokenType.GT)
            case '"' | "'":
                self.string(c)
    
    KEYWORDS = {
        # Control flow
        "IF": TokenType.IF,
        "ELSE": TokenType.ELSE,
        "ENDIF": TokenType.ENDIF,
        "FOR": TokenType.FOR,
        "EACH": TokenType.EACH,
        "IN": TokenType.IN,
        "ENDFOR": TokenType.ENDFOR,
        
        # Components
        "COMPONENT": TokenType.COMPONENT,
        "END_COMPONENT": TokenType.END_COMPONENT,
        "FORM": TokenType.FORM,
        "END_FORM": TokenType.END_FORM,
        
        # UI Elements
        "TITLE": TokenType.TITLE,
        "INPUT": TokenType.INPUT,
        "CHECKBOX": TokenType.CHECKBOX,
        "QUICKPICK": TokenType.QUICKPICK,
        "ICON": TokenType.ICON,
        "ITEM_LIST": TokenType.ITEM_LIST,
        "CREATE_TABLE": TokenType.CREATE_TABLE,
        "NAVBAR": TokenType.NAVBAR,
        "CHART": TokenType.CHART,
        
        # Data
        "SAVE_DATA": TokenType.SAVE_DATA,
        "LOAD_DATA": TokenType.LOAD_DATA,
        "DEFINE_VAR": TokenType.DEFINE_VAR
    }
    
    def identifier(self):
        # Read until we hit a non-identifier character
        while self.peek().isalnum() or self.peek() == '_':
            self.advance()
            
        # Special handling for two-word commands
        text = self.source[self.start:self.current]
        if text.upper() in ["ITEM", "CREATE"]:
            # Look ahead for "LIST" or "TABLE"
            saved_current = self.current
            saved_column = self.column
            
            while self.peek().isspace() and self.peek() != '\n':
                self.advance()
                
            next_word_start = self.current
            while self.peek().isalnum():
                self.advance()
                
            next_word = self.source[next_word_start:self.current]
            combined = f"{text}_{next_word}".upper()
            
            if combined in self.KEYWORDS:
                return self.add_token(self.KEYWORDS[combined])
                
            # Backtrack if not a valid two-word command
            self.current = saved_current
            self.column = saved_column
            
        return self.add_token(self.KEYWORDS.get(text.upper(), TokenType.IDENTIFIER), text)
    
    def string(self, quote: str):
        while self.peek() != quote and not self.is_at_end():
            if self.peek() == '\n':
                self.line += 1
                self.column = 1
            self.advance()
            
        if self.is_at_end():
            raise SyntaxError(f"Unterminated string at line {self.line}")
            
        # Consume closing quote
        self.advance()
        value = self.source[self.start + 1:self.current - 1]
        self.add_token(TokenType.STRING, value)

    def number(self):
        while self.peek().isdigit():
            self.advance()
            
        if self.peek() == '.' and self.peek_next().isdigit():
            self.advance()  # Consume '.'
            while self.peek().isdigit():
                self.advance()
                
        self.add_token(TokenType.NUMBER, float(self.source[self.start:self.current]))
    
    def add_token(self, type: TokenType, literal: Any = None):
        text = self.source[self.start:self.current]
        self.tokens.append(Token(type, text, literal, self.line, self.column))
        self.column += self.current - self.start
        
    def is_at_end(self) -> bool:
        return self.current >= len(self.source)
        
    def advance(self) -> str:
        self.current += 1
        return self.source[self.current - 1]
        
    def peek(self) -> str:
        if self.is_at_end():
            return '\0'
        return self.source[self.current]
        
    def peek_next(self) -> str:
        if self.current + 1 >= len(self.source):
            return '\0'
        return self.source[self.current + 1]

class DevCarDataStore:
    """Simple JSON-based data storage system"""
    def __init__(self, base_path=None):
        self.base_path = base_path or os.path.join(os.getcwd(), 'devcar_data')
        os.makedirs(self.base_path, exist_ok=True)
    
    def _get_collection_path(self, collection):
        return os.path.join(self.base_path, f"{collection}.json")
    
    def save_data(self, collection, data):
        """Save data to a collection"""
        path = self._get_collection_path(collection)
        current_data = self.load_data(collection) or []
        if isinstance(current_data, list):
            current_data.append(data)
        else:
            current_data = [current_data, data]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=2, ensure_ascii=False)
    
    def load_data(self, collection, query=None):
        """Load data from a collection with optional query"""
        path = self._get_collection_path(collection)
        if not os.path.exists(path):
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if query:
            # Simple query format: field:value
            field, value = query.split(':')
            return [item for item in data if str(item.get(field)) == value]
        return data

class DevCarStateManager:
    """Manages application state and variables"""
    def __init__(self):
        self.state = {}
        self.watchers = {}
    
    def define_var(self, name, initial_value):
        self.state[name] = initial_value
        
    def get_var(self, name):
        return self.state.get(name)
        
    def set_var(self, name, value):
        self.state[name] = value
        # Notify watchers
        if name in self.watchers:
            for callback in self.watchers[name]:
                callback(value)
    
    def watch_var(self, name, callback):
        if name not in self.watchers:
            self.watchers[name] = []
        self.watchers[name].append(callback)

# --- DevCAR Language Core ---

class ASTNode:
    def accept(self, visitor):
        method_name = f'visit_{self.__class__.__name__}'
        visitor_method = getattr(visitor, method_name)
        return visitor_method(self)

class Expression(ASTNode):
    pass

class Statement(ASTNode):
    pass

@dataclass
class Literal(Expression):
    value: Any
    
@dataclass
class Variable(Expression):
    name: str
    
@dataclass
class PropertyAccess(Expression):
    object: Expression
    property: str
    
@dataclass
class BinaryOp(Expression):
    left: Expression
    operator: str
    right: Expression
    
@dataclass
class IfStatement(Statement):
    condition: Expression
    then_branch: List[Statement]
    else_branch: Optional[List[Statement]] = None

@dataclass
class ForEachStatement(Statement):
    item_var: str
    collection: Expression
    body: List[Statement]
    
@dataclass
class ComponentStatement(Statement):
    name: str
    properties: dict
    body: List[Statement]
    
@dataclass
class FormStatement(Statement):
    name: str
    action: Optional[str]
    body: List[Statement]

@dataclass
class TitleStatement(Statement):
    content: str

@dataclass
class InputStatement(Statement):
    name: str
    type: str
    placeholder: Optional[str]

@dataclass
class CheckboxStatement(Statement):
    name: str
    default: bool

@dataclass
class QuickPickStatement(Statement):
    label: str
    style: str

@dataclass
class IconStatement(Statement):
    name: str
    css_class: Optional[str]

@dataclass
class ItemListStatement(Statement):
    items: List[Expression]

@dataclass
class TableStatement(Statement):
    title: str
    columns: List[str]

@dataclass
class NavbarStatement(Statement):
    links: List[str]
    icon: Optional[str]

@dataclass
class ChartStatement(Statement):
    title: str
    type: str
    data: List[float]

@dataclass
class SaveDataStatement(Statement):
    collection: str
    data: Expression

@dataclass
class LoadDataStatement(Statement):
    collection: str
    query: Optional[Expression]

@dataclass
class DefineVarStatement(Statement):
    name: str
    initial_value: Expression

@dataclass
class Diagnostic:
    line: int
    column: int
    message: str
    severity: str = "error"  # or "warning"
    code: Optional[str] = None

class ParseError(Exception):
    def __init__(self, token: Token, message: str, code: Optional[str] = None):
        self.token = token
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {message} at line {token.line}, column {token.column}")

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
        self.diagnostics: List[Diagnostic] = []
        self.control_stack: List[str] = []  # Track nested control structures
        
    def parse(self) -> List[Statement]:
        statements = []
        while not self.is_at_end():
            statements.append(self.statement())
        return statements
    
    def statement(self) -> Statement:
        if self.match(TokenType.IF):
            return self.if_statement()
        if self.match(TokenType.FOR):
            return self.for_each_statement()
        if self.match(TokenType.COMPONENT):
            return self.component_statement()
        if self.match(TokenType.FORM):
            return self.form_statement()
        if self.match(TokenType.TITLE):
            return self.title_statement()
        if self.match(TokenType.INPUT):
            return self.input_statement()
        if self.match(TokenType.CHECKBOX):
            return self.checkbox_statement()
        if self.match(TokenType.QUICKPICK):
            return self.quickpick_statement()
        if self.match(TokenType.ICON):
            return self.icon_statement()
        if self.match(TokenType.ITEM_LIST):
            return self.item_list_statement()
        if self.match(TokenType.CREATE_TABLE):
            return self.table_statement()
        if self.match(TokenType.NAVBAR):
            return self.navbar_statement()
        if self.match(TokenType.CHART):
            return self.chart_statement()
        if self.match(TokenType.SAVE_DATA):
            return self.save_data_statement()
        if self.match(TokenType.LOAD_DATA):
            return self.load_data_statement()
        if self.match(TokenType.DEFINE_VAR):
            return self.define_var_statement()
            
        raise SyntaxError(f"Unexpected token {self.peek().type} at line {self.peek().line}")
        
    def for_each_statement(self) -> ForEachStatement:
        self.consume(TokenType.EACH, "Expect 'EACH' after 'FOR'")
        self.consume(TokenType.LBRACKET, "Expect '[' after 'FOR EACH'")
        
        item_var = self.consume(TokenType.IDENTIFIER, "Expect item variable name").lexeme
        self.consume(TokenType.IN, "Expect 'IN' after item variable")
        collection = self.expression()
        
        self.consume(TokenType.RBRACKET, "Expect ']' after collection expression")
        
        body = []
        while not self.check(TokenType.ENDFOR) and not self.is_at_end():
            body.append(self.statement())
            
        self.consume(TokenType.ENDFOR, "Expect 'ENDFOR' after for loop body")
        return ForEachStatement(item_var, collection, body)
        
    def component_statement(self) -> ComponentStatement:
        self.consume(TokenType.LBRACKET, "Expect '[' after 'COMPONENT'")
        properties = self.parse_properties()
        self.consume(TokenType.RBRACKET, "Expect ']' after component properties")
        
        body = []
        while not self.check(TokenType.END_COMPONENT) and not self.is_at_end():
            body.append(self.statement())
            
        self.consume(TokenType.END_COMPONENT, "Expect 'END_COMPONENT' after component body")
        return ComponentStatement(properties.get("name", ""), properties, body)
        
    def form_statement(self) -> FormStatement:
        self.consume(TokenType.LBRACKET, "Expect '[' after 'FORM'")
        name = self.consume(TokenType.IDENTIFIER, "Expect form name").lexeme
        action = None
        
        if self.match(TokenType.COMMA):
            self.consume(TokenType.IDENTIFIER, "Expect 'action' property")
            self.consume(TokenType.EQUALS, "Expect '=' after 'action'")
            action = self.consume(TokenType.IDENTIFIER, "Expect action name").lexeme
            
        self.consume(TokenType.RBRACKET, "Expect ']' after form properties")
        
        body = []
        while not self.check(TokenType.END_FORM) and not self.is_at_end():
            body.append(self.statement())
            
        self.consume(TokenType.END_FORM, "Expect 'END_FORM' after form body")
        return FormStatement(name, action, body)
        
    def parse_properties(self) -> Dict[str, Any]:
        properties = {}
        while not self.check(TokenType.RBRACKET):
            name = self.consume(TokenType.IDENTIFIER, "Expect property name").lexeme
            self.consume(TokenType.EQUALS, "Expect '=' after property name")
            value = self.expression()
            properties[name] = value
            
            if not self.check(TokenType.RBRACKET):
                self.consume(TokenType.COMMA, "Expect ',' between properties")
        return properties
        
    def title_statement(self) -> TitleStatement:
        """Parse TITLE[content]"""
        self.consume(TokenType.LBRACKET, "Expect '[' after 'TITLE'")
        content = self.expression()
        self.consume(TokenType.RBRACKET, "Expect ']' after title content")
        return TitleStatement(content)
        
    def input_statement(self) -> InputStatement:
        """Parse INPUT[name, type=type, placeholder=text]"""
        self.consume(TokenType.LBRACKET, "Expect '[' after 'INPUT'")
        name = self.consume(TokenType.IDENTIFIER, "Expect input name").lexeme
        
        type = "text"  # default
        placeholder = None
        
        if self.match(TokenType.COMMA):
            properties = self.parse_properties()
            type = properties.get("type", type)
            placeholder = properties.get("placeholder")
            
        self.consume(TokenType.RBRACKET, "Expect ']' after input properties")
        return InputStatement(name, type, placeholder)
        
    def checkbox_statement(self) -> CheckboxStatement:
        """Parse CHECKBOX[name, default=true|false]"""
        self.consume(TokenType.LBRACKET, "Expect '[' after 'CHECKBOX'")
        name = self.consume(TokenType.IDENTIFIER, "Expect checkbox name").lexeme
        
        default = False
        if self.match(TokenType.COMMA):
            properties = self.parse_properties()
            default_str = str(properties.get("default", "false")).lower()
            default = default_str == "true"
            
        self.consume(TokenType.RBRACKET, "Expect ']' after checkbox properties")
        return CheckboxStatement(name, default)
        
    def quickpick_statement(self) -> QuickPickStatement:
        """Parse QUICKPICK[label, style=primary|danger|default]"""
        self.consume(TokenType.LBRACKET, "Expect '[' after 'QUICKPICK'")
        label = self.expression()
        style = "primary"
        
        if self.match(TokenType.COMMA):
            properties = self.parse_properties()
            style = properties.get("style", style)
            
        self.consume(TokenType.RBRACKET, "Expect ']' after quickpick properties")
        return QuickPickStatement(label, style)
        
    def icon_statement(self) -> IconStatement:
        """Parse ICON[name, class=css_class]"""
        self.consume(TokenType.LBRACKET, "Expect '[' after 'ICON'")
        name = self.expression()
        css_class = None
        
        if self.match(TokenType.COMMA):
            properties = self.parse_properties()
            css_class = properties.get("class")
            
        self.consume(TokenType.RBRACKET, "Expect ']' after icon properties")
        return IconStatement(name, css_class)
        
    def item_list_statement(self) -> ItemListStatement:
        """Parse ITEM_LIST[item1, item2, ...]"""
        self.consume(TokenType.LBRACKET, "Expect '[' after 'ITEM_LIST'")
        items = []
        
        if not self.check(TokenType.RBRACKET):
            items.append(self.expression())
            while self.match(TokenType.COMMA):
                items.append(self.expression())
                
        self.consume(TokenType.RBRACKET, "Expect ']' after list items")
        return ItemListStatement(items)
        
    def table_statement(self) -> TableStatement:
        """Parse CREATE_TABLE[title, col1, col2, ...]"""
        self.consume(TokenType.LBRACKET, "Expect '[' after 'CREATE_TABLE'")
        title = self.consume(TokenType.IDENTIFIER, "Expect table title").lexeme
        columns = []
        
        while self.match(TokenType.COMMA):
            columns.append(self.consume(TokenType.IDENTIFIER, "Expect column name").lexeme)
            
        self.consume(TokenType.RBRACKET, "Expect ']' after table columns")
        return TableStatement(title, columns)
        
    def navbar_statement(self) -> NavbarStatement:
        """Parse NAVBAR[Links=link1 link2, Icon=icon]"""
        self.consume(TokenType.LBRACKET, "Expect '[' after 'NAVBAR'")
        properties = self.parse_properties()
        
        links = []
        if "Links" in properties:
            links = str(properties["Links"]).split()
            
        icon = properties.get("Icon")
        self.consume(TokenType.RBRACKET, "Expect ']' after navbar properties")
        return NavbarStatement(links, icon)
        
    def chart_statement(self) -> ChartStatement:
        """Parse CHART[title, type=type, data=val1 val2 ...]"""
        self.consume(TokenType.LBRACKET, "Expect '[' after 'CHART'")
        title = self.expression()
        type = "bar"  # default
        data = []
        
        if self.match(TokenType.COMMA):
            properties = self.parse_properties()
            type = properties.get("type", type)
            if "data" in properties:
                data = [float(x) for x in str(properties["data"]).split()]
                
        self.consume(TokenType.RBRACKET, "Expect ']' after chart properties")
        return ChartStatement(title, type, data)
        
    def if_statement(self) -> IfStatement:
        self.consume(TokenType.LBRACKET, "Expect '[' after 'IF'")
        condition = self.expression()
        self.consume(TokenType.RBRACKET, "Expect ']' after condition")
        
        then_branch = []
        while not self.check(TokenType.ENDIF) and not self.check(TokenType.ELSE):
            then_branch.append(self.statement())
            
        else_branch = None
        if self.match(TokenType.ELSE):
            else_branch = []
            while not self.check(TokenType.ENDIF):
                else_branch.append(self.statement())
                
        self.consume(TokenType.ENDIF, "Expect 'ENDIF' after if statement")
        return IfStatement(condition, then_branch, else_branch)
    
    def expression(self) -> Expression:
        return self.equality()
    
    def equality(self) -> Expression:
        expr = self.primary()
        
        while self.match(TokenType.EQUALS):
            operator = self.previous().lexeme
            right = self.primary()
            expr = BinaryOp(expr, operator, right)
            
        return expr
    
    def primary(self) -> Expression:
        if self.match(TokenType.NUMBER, TokenType.STRING):
            return Literal(self.previous().literal)
            
        if self.match(TokenType.IDENTIFIER):
            expr = Variable(self.previous().lexeme)
            while self.match(TokenType.DOT):
                name = self.consume(TokenType.IDENTIFIER, "Expect property name after '.'")
                expr = PropertyAccess(expr, name.lexeme)
            return expr
            
        raise SyntaxError(f"Unexpected token: {self.peek()}")
        
    def match(self, *types) -> bool:
        for type in types:
            if self.check(type):
                self.advance()
                return True
        return False
        
    def check(self, type) -> bool:
        if self.is_at_end():
            return False
        return self.peek().type == type
        
    def advance(self) -> Token:
        if not self.is_at_end():
            self.current += 1
        return self.previous()
        
    def is_at_end(self) -> bool:
        return self.peek().type == TokenType.EOF
        
    def peek(self) -> Token:
        return self.tokens[self.current]
        
    def previous(self) -> Token:
        return self.tokens[self.current - 1]
        
    def consume(self, type: TokenType, message: str) -> Token:
        if self.check(type):
            return self.advance()
        raise SyntaxError(f"{message} at line {self.peek().line}")

class HTMLEscaper:
    @staticmethod
    def escape(text: str) -> str:
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

class HTMLGenerator:
    def __init__(self, context: dict):
        self.context = context
        self.current_form = None
        self.loop_stack = []
        self.escaper = HTMLEscaper()
        self.chart_included = False
        
    def generate(self, statements: List[Statement]) -> str:
        body_content = '\n'.join(stmt.accept(self) for stmt in statements)
        return self._wrap_html(body_content)
        
    def _wrap_html(self, body_content: str) -> str:
        """Wrap content in a complete HTML document with proper headers and scripts."""
        # Use strict CSP but allow the specific external CDNs for scripts/styles; also provide local fallback CSS
        chart_script = '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>' if self.chart_included else ''
        return f"""<!DOCTYPE html>
        <html lang=\"en"> 
        <head>
            <meta charset=\"UTF-8">
            <meta name=\"viewport" content=\"width=device-width, initial-scale=1.0">
            <title>DevCAR Application</title>
            <meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'self'; script-src 'self' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; script-src-elem 'self' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; style-src 'self' https://cdn.tailwindcss.com https://cdnjs.cloudflare.com; style-src-elem 'self' https://cdn.tailwindcss.com https://cdnjs.cloudflare.com;">
            <!-- Prefer CDN resources but include local fallback -->
            <script src=\"https://cdn.tailwindcss.com\"></script>
            {chart_script}
            <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
            <link rel=\"stylesheet\" href=\"file:///{os.path.abspath('preview_fallback.css')}"> 
            <script>
                // Minimal bindings and update loop (kept inline for simplicity)
                window.devcarBindings = {{}};
                const updateBindings = () => {{
                    document.querySelectorAll('[data-bind]').forEach(el => {{
                        const key = el.getAttribute('data-bind');
                        const value = window.devcarBindings[key];
                        if (value !== undefined) {{ el.textContent = value; }}
                    }});
                }};
                document.addEventListener('DOMContentLoaded', () => {{ updateBindings(); new MutationObserver(updateBindings).observe(document.body, {{ childList: true, subtree: true }}); }});
            </script>
        </head>
        <body>
            <div id=\"app-container">{body_content}</div>
        </body>
        </html>"""
        
    def visit_IfStatement(self, node: IfStatement) -> str:
        condition_result = self.evaluate_expression(node.condition)
        if condition_result:
            return '\n'.join(stmt.accept(self) for stmt in node.then_branch)
        elif node.else_branch:
            return '\n'.join(stmt.accept(self) for stmt in node.else_branch)
        return ''
        
    def visit_ForEachStatement(self, node: ForEachStatement) -> str:
        collection = self.evaluate_expression(node.collection)
        if not collection:
            return ''
            
        result = []
        for item in collection:
            # Create new scope with item variable
            with self.scope_with(node.item_var, item):
                result.extend(stmt.accept(self) for stmt in node.body)
        return '\n'.join(result)
        
    def visit_ComponentStatement(self, node: ComponentStatement) -> str:
        props = {k: self.evaluate_expression(v) for k, v in node.properties.items()}
        css_class = props.get('class', '')
        
        html = f'<div class="bg-white rounded-lg shadow-lg overflow-hidden {css_class} mb-6">'
        html += '<div class="px-6 py-4">'
        if 'title' in props:
            html += f'<div class="font-bold text-xl mb-2">{self.escaper.escape(str(props["title"]))}</div>'
        html += '<div class="card-content">'
        
        # Process component body
        html += '\n'.join(stmt.accept(self) for stmt in node.body)
        
        html += '</div></div></div>'
        return html
        
    def visit_FormStatement(self, node: FormStatement) -> str:
        self.current_form = node.name
        
        html = f'''
        <form id="{self.escaper.escape(node.name)}" 
              class="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4"
              onsubmit="event.preventDefault(); 
                       console.log('Form submitted:', window.devcarBindings);">
        '''
        
        # Process form body
        html += '\n'.join(stmt.accept(self) for stmt in node.body)
        
        html += '</form>'
        self.current_form = None
        return html

    # --- Basic statement visitors (rendering primitives) ---
    def visit_TitleStatement(self, node: TitleStatement) -> str:
        content = self.escaper.escape(str(node.content))
        return f'<h2 class="text-xl font-bold mb-2">{content}</h2>'

    def visit_InputStatement(self, node: InputStatement) -> str:
        name = self.escaper.escape(node.name)
        placeholder = self.escaper.escape(node.placeholder or '')
        typ = 'text' if node.type is None else self.escaper.escape(node.type)
        return f'<div class="mb-3"><label class="block text-sm font-medium">{name}</label><input type="{typ}" name="{name}" placeholder="{placeholder}" class="p-2 border rounded w-full" /></div>'

    def visit_CheckboxStatement(self, node: CheckboxStatement) -> str:
        name = self.escaper.escape(node.name)
        checked = 'checked' if node.default else ''
        return f'<div class="mb-3"><label><input type="checkbox" name="{name}" {checked}/> {name}</label></div>'

    def visit_ItemListStatement(self, node: ItemListStatement) -> str:
        items = []
        for item_expr in node.items:
            try:
                val = self.evaluate_expression(item_expr)
            except Exception:
                val = str(item_expr)
            items.append(self.escaper.escape(str(val)))
        list_items = ''.join(f'<li class="flex items-center"><span class="icon-check"></span><span>{it}</span></li>' for it in items)
        return f'<ul class="mb-4">{list_items}</ul>'

    def visit_TableStatement(self, node: TableStatement) -> str:
        cols = ''.join(f'<th class="px-4 py-2">{self.escaper.escape(c)}</th>' for c in node.columns)
        sample_row = ''.join(f'<td class="px-4 py-2">Sample</td>' for _ in node.columns)
        return f'<div class="mb-6"><h3 class="font-bold">{self.escaper.escape(node.title)}</h3><table class="table"><thead><tr>{cols}</tr></thead><tbody><tr>{sample_row}</tr></tbody></table></div>'

    def visit_NavbarStatement(self, node: NavbarStatement) -> str:
        links_html = ''.join(f'<a href="#" class="mr-4">{self.escaper.escape(link)}</a>' for link in node.links)
        return f'<nav class="mb-4">{links_html}</nav>'
        
    def evaluate_expression(self, expr: Expression) -> Any:
        if isinstance(expr, Literal):
            return expr.value
            
        if isinstance(expr, Variable):
            return self.context.get(expr.name)
            
        if isinstance(expr, PropertyAccess):
            obj = self.evaluate_expression(expr.object)
            if obj is None:
                return None
            return obj.get(expr.property)
            
        if isinstance(expr, BinaryOp):
            left = self.evaluate_expression(expr.left)
            right = self.evaluate_expression(expr.right)
            
            if expr.operator == '==':
                return left == right
            # Add other operators
            
        return None
        
    @contextmanager
    def scope_with(self, name: str, value: Any):
        """Context manager for temporary variable scope"""
        old_value = self.context.get(name)
        self.context[name] = value
        try:
            yield
        finally:
            if old_value is not None:
                self.context[name] = old_value
            else:
                del self.context[name]

class DevCarContext:
    """Manages variables and state during execution"""
    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.components = {}
    
    def set_var(self, name, value):
        self.variables[name] = value
    
    def get_var(self, name):
        return self.variables.get(name)

# Legacy parse_condition removed in favor of proper Parser

def get_app_settings():
    """Returns basic configuration for the application."""
    return {
        "title": "DevCAR Studio (Web Edition)",
        "font_family": "Consolas, monospace",
        "font_size": 12,
        "php_path": "php",
        "node_path": "node"
    }

# --- THE DEV CAR STUDIO IDE CLASS (VS CODE LOOK) ---

class DevCarIDE(QMainWindow):
    """
    A single-page IDE interface designed to resemble VS Code.
    No changes needed here for the front-end rendering update.
    """
    def __init__(self, filepath=None):
        super().__init__()
        self.settings = get_app_settings()
        self.setWindowTitle(f"{self.settings['title']} - IDE Mode")
        self.setGeometry(100, 100, 1200, 800)

        self.current_filepath = filepath
        self._load_file(filepath)

        self._setup_ui()

    def _load_file(self, filepath):
        """Loads file content into the editor and sets the current path."""
        self.content = ""
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    self.content = f.read()
                self.current_filepath = filepath
                self.setWindowTitle(f"{self.settings['title']} - IDE Mode: {os.path.basename(filepath)}")
            except Exception as e:
                self.content = f"# Error loading file: {e}"
                self.current_filepath = None
        else:
            self.content = "## Welcome to DevCAR IDE\n# Start high-level coding here."
            if not self.current_filepath:
                 # Default to a .dev file in the current directory if we start fresh
                 self.current_filepath = os.path.join(os.getcwd(), "untitled.dev")
                 self.setWindowTitle(f"{self.settings['title']} - IDE Mode: untitled.dev")


    def _setup_ui(self):
        """Builds the VS Code-like UI layout."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Sidebar (File/Tool Explorer - VS Code Look)
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet("background-color: #252526; color: #CCCCCC;")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(5, 10, 5, 10)
        sidebar_layout.addWidget(QListWidget()) # Placeholder for file explorer
        main_layout.addWidget(self.sidebar)

        # 2. Editor and Terminal/Output Area
        editor_area = QWidget()
        editor_layout = QVBoxLayout(editor_area)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)

        # Tab Widget for Editor (Main Content)
        self.editor = QTextEdit()
        self.editor.setFont(QFont(self.settings["font_family"], self.settings["font_size"]))
        self.editor.setText(self.content)
        self.editor.setStyleSheet("background-color: #1E1E1E; color: #D4D4D4; padding: 10px;")

        # Tab Widget for Quick Config/Buttons
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet('''
            QTabWidget::pane { border: 0; }
            QTabBar::tab { background: #3C3C3C; color: #CCCCCC; padding: 8px 15px; }
            QTabBar::tab:selected { background: #1E1E1E; }
        ''')
        self.tabs.addTab(self.editor, "Code Editor")
        self.tabs.addTab(self._create_quick_config_tab(), "Quick Configs")
        editor_layout.addWidget(self.tabs, 3) # Editor takes 3/4 space

        # Output/Terminal (Bottom Panel)
        self.output_terminal = QTextEdit()
        self.output_terminal.setReadOnly(True)
        self.output_terminal.setFixedHeight(150)
        self.output_terminal.setFont(QFont(self.settings["font_family"], 10))
        self.output_terminal.setStyleSheet("background-color: #000000; color: #00FF00; padding: 5px;")
        self.output_terminal.setText("Terminal ready. DevCAR IDE v2.0\n")
        editor_layout.addWidget(self.output_terminal, 1) # Terminal takes 1/4 space

        main_layout.addWidget(editor_area)

        # 3. Status Bar
        self.setStatusBar(QStatusBar())
        self.statusBar().setStyleSheet("background-color: #007ACC; color: white;")
        self.statusBar().showMessage(f"File: {os.path.basename(self.current_filepath) if self.current_filepath else 'New File'} | Ready")

        self._create_toolbar()
        self._create_menubar()

    def _create_menubar(self):
        """Creates the File, Edit, Run menu actions."""
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)

        file_menu = menu_bar.addMenu("&File")
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        run_menu = menu_bar.addMenu("&Run")
        run_devcar = QAction("Run DevCAR (.dev)", self)
        run_devcar.triggered.connect(lambda: self.run_script('dev'))
        run_menu.addAction(run_devcar)

        compile_menu = menu_bar.addMenu("&Compile")
        compile_devc = QAction("Compile to .devc (Simple Obfuscation)", self)
        compile_devc.triggered.connect(self.compile_devc)
        compile_menu.addAction(compile_devc)

    def _create_toolbar(self):
        """Creates a simplified toolbar with common actions."""
        tool_bar = QToolBar("Main Toolbar")
        tool_bar.setStyleSheet("background-color: #3C3C3C;")
        self.addToolBar(tool_bar)

        tool_bar.addAction(QIcon.fromTheme("document-save"), "Save", self.save_file)
        # Added a dedicated Preview button
        tool_bar.addAction(QIcon.fromTheme("system-run"), "Preview in Launcher", lambda: self.run_script('dev'))
        tool_bar.addAction(QIcon.fromTheme("object-unlocked"), "Compile", self.compile_devc)

    def _create_quick_config_tab(self):
        """Creates the 'Quick Configs' tab with buttons for tables, columns, etc."""
        config_widget = QWidget()
        grid = QGridLayout(config_widget)
        grid.setSpacing(10)
        grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        row = 0
        col = 0
        buttons = [
            ("Insert Table (3x2)", "CREATE TABLE[Users, Name, Age]"),
            ("Insert Item List", "ITEM LIST[Apple, Banana, Cherry]"),
            ("Set Layout: Grid 2 Columns", "LAYOUT[grid, columns=2]"),
            ("Quick Pick Button: Submit", "QUICKPICK[Submit, primary]"),
            ("Quick Pick Button: Danger", "QUICKPICK[Delete, danger]"),
            ("Include HTML/CSS (Tailwind)", "HTML_BLOCK[<p class='text-xl'>Custom HTML</p>]"),
            ("Include AwesomeFont Icon", "ICON[fa-rocket, text-indigo-500]"),
            ("Run PHP Block", "PHP_BLOCK[echo 'Server side content']"),
            ("Run Python Logic", "PYTHON_LOGIC[print('Backend data')]"),
        ]

        for text, code_snippet in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet('''
                QPushButton {
                    background-color: #007ACC; color: white; border: none; padding: 10px;
                    border-radius: 5px; text-align: left;
                }
                QPushButton:hover { background-color: #005A99; }
            ''')
            btn.clicked.connect(lambda _, snippet=code_snippet: self._insert_snippet(snippet))
            grid.addWidget(btn, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1

        return config_widget

    def _insert_snippet(self, snippet):
        """Inserts a custom DevCAR code snippet into the editor."""
        cursor = self.editor.textCursor()
        cursor.insertText(f"\n{snippet}\n")
        self.output_terminal.append(f"-> Inserted: {snippet.split('[')[0]} snippet.")

    def save_file(self):
        """Saves the current content back to the file."""
        if not self.current_filepath:
            self.output_terminal.append("Error: Cannot save. Filepath is unknown.")
            return

        try:
            # Save using UTF-8 to avoid Windows code page errors for special characters
            with open(self.current_filepath, 'w', encoding='utf-8') as f:
                f.write(self.editor.toPlainText())
            self.statusBar().showMessage(f"File saved successfully: {self.current_filepath}", 3000)
            self.output_terminal.append(f"File saved: {self.current_filepath}")
        except Exception as e:
            self.output_terminal.append(f"Error saving file: {e}")
            self.statusBar().showMessage("Error saving file.", 3000)

    def compile_devc(self):
        """Compile source to .devc bytecode format."""
        if not self.current_filepath:
            QMessageBox.warning(self, "Compilation Error", "Please save the file first.")
            return
            
        try:
            # Read using UTF-8 and replace invalid characters if present
            with open(self.current_filepath, 'r', encoding='utf-8', errors='replace') as f:
                source = f.read()
                
            # Use the real parser/compiler
            tokens = DevCarLexer(source).lex()
            ast = Parser(tokens).parse()
            
            # Generate bytecode (simple for now)
            bytecode = self._generate_bytecode(ast)
            
            output_path = os.path.splitext(self.current_filepath)[0] + '.devc'
            with open(output_path, 'wb') as f:
                f.write(bytecode)
                
            self.output_terminal.append(f"Compiled successfully to {output_path}")
            self.statusBar().showMessage("Compilation successful", 3000)
            
        except Exception as e:
            QMessageBox.critical(self, "Compilation Error", str(e))
            
    def _generate_bytecode(self, ast: List[Statement]) -> bytes:
        """Simple bytecode generation."""
        # Basic bytecode format for now:
        # 1. Magic number (4 bytes)
        # 2. Version (1 byte)
        # 3. Serialized AST with basic op codes
        magic = b'DEVC'
        version = b'\x01'
        
        # Serialize AST nodes to simple bytecode
        # This is a basic implementation that will be enhanced later
        def serialize_node(node: ASTNode) -> bytes:
            if isinstance(node, IfStatement):
                return b'\x01' + serialize_node(node.condition)
            elif isinstance(node, ForEachStatement):
                return b'\x02' + node.item_var.encode()
            # Add more node types...
            return b'\x00'  # NOP for unknown nodes
            
        body = b''.join(serialize_node(stmt) for stmt in ast)
        return magic + version + body


    def run_script(self, lang_type):
        """
        Simulates running the script. In IDE mode, this saves the file and
        then relaunches itself in Launcher mode for viewing the generated UI.
        """
        self.save_file() # Ensure latest changes are saved

        # Relaunch the application in Launcher mode to view the generated UI
        # This is the simplest way to simulate the "double-click" action.
        QProcess.startDetached(sys.executable, [os.path.abspath(__file__), self.current_filepath])
        self.output_terminal.append(f"Launching DevCAR Launcher to preview {os.path.basename(self.current_filepath)}...")

# --- THE DEV CAR LAUNCHER CLASS (RUNTIME/EXECUTION) ---

class DevCarLauncher(QMainWindow):
    """
    The custom windowed application launcher.
    Handles execution of DevCAR files by rendering a dynamic front-end using QWebEngineView.
    """
    def __init__(self, filepath):
        super().__init__()
        self.settings = get_app_settings()
        self.filepath = filepath
        self.setWindowTitle(f"{self.settings['title']} - Launcher Mode: {os.path.basename(filepath)}")
        self.setGeometry(200, 200, 1000, 750)
        self.setStyleSheet("background-color: #2D2D2D;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 1. Path/Status Bar
        self.status_bar_widget = QLineEdit(f"File: {filepath} | Status: Ready")
        self.status_bar_widget.setReadOnly(True)
        self.status_bar_widget.setStyleSheet("background-color: #1E1E1E; color: #34D399; padding: 5px; border: none;")
        self.layout.addWidget(self.status_bar_widget)

        # 2. THE HIGH-LEVEL FRONT-END RENDERER (QWebEngineView)
        self.web_view = QWebEngineView()
        self.layout.addWidget(self.web_view)

        # 3. Execution Control/Relaunch Button
        self.run_button = QPushButton("Relaunch DevCAR App")
        self.run_button.setStyleSheet('''
            QPushButton {
                background-color: #34D399; color: black; border: none; padding: 10px;
                font-weight: bold; border-radius: 5px;
            }
            QPushButton:hover { background-color: #10B981; }
        ''')
        self.run_button.clicked.connect(self.execute_file)
        self.layout.addWidget(self.run_button)

        self.execute_file() # Auto-run on launch

    def execute_file(self):
        """Determines the file type and executes/renders it."""
        filepath = self.filepath
        if not os.path.exists(filepath):
            self.status_bar_widget.setText(f"[ERROR] File not found: {filepath}")
            return

        base, ext = os.path.splitext(filepath)
        self.status_bar_widget.setText(f"File: {filepath} | Status: Rendering {ext.upper()}...")

        if ext == '.devc':
            # Load compiled bytecode
            try:
                with open(filepath, 'rb') as f:
                    if f.read(4) != b'DEVC':
                        raise ValueError("Invalid .devc file format")
                    version = f.read(1)[0]
                    if version != 1:
                        raise ValueError(f"Unsupported .devc version: {version}")
                    # Execute bytecode (to be implemented)
                    self.status_bar_widget.setText(f"File: {filepath} | Status: Executing compiled bytecode")
            except Exception as e:
                self.status_bar_widget.setText(f"[ERROR] Failed to load .devc file: {e}")
                return
                
        elif ext in ('.dev', '.devkpi', '.devsdk'):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    source = f.read()
                
                # Parse with AST
                tokens = DevCarLexer(source).lex()
                ast = Parser(tokens).parse()
                
                # Initialize context
                context = {
                    'user': {'status': 'active', 'is_admin': True},
                    'shopping_cart': []
                }
                
                # Generate HTML via AST visitor
                try:
                    generator = HTMLGenerator(context)
                    html_content = generator.generate(ast)
                    self.web_view.setHtml(html_content, QUrl("file:///"))
                    self.status_bar_widget.setText(f"File: {filepath} | Status: Rendered DevCAR UI (AST-based)")
                except Exception as e:
                    # Render an error surface in the webview so user isn't left with a blank page
                    err_html = f'''
                    <div style=\"background:#fff3f2;border:1px solid #fca5a5;padding:16px;border-radius:8px;margin:16px;">
                        <h2 style=\"color:#991b1b;">Render Error</h2>
                        <pre style=\"white-space:pre-wrap;">{self.escaper.escape(str(e))}\n{self.escaper.escape(traceback.format_exc())}</pre>
                    </div>
                    '''
                    self.web_view.setHtml(err_html, QUrl("file:///"))
                    self.status_bar_widget.setText(f"File: {filepath} | Status: Render error: {e}")
            except Exception as e:
                self.status_bar_widget.setText(f"[ERROR] Failed to parse/render file: {e}")
                return
        elif ext == '.py' or ext == '.php' or ext == '.js':
             # For raw language files, we just display the code as text in the web view
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    code_content = f.read()
                # Use HTML pre tag for monospaced code viewing
                html_content = f'''
                <body style="background-color: #2D2D2D; color: #D4D4D4; font-family: monospace;">
                    <h2>Raw {ext.upper()} Code Execution Placeholder</h2>
                    <pre style="background-color: #1E1E1E; padding: 15px; border-radius: 5px;">{code_content}</pre>
                    <p>Execution of {ext.upper()} code would run via subprocess (terminal) and not render a UI.</p>
                </body>
                '''
                self.web_view.setHtml(html_content, QUrl("file:///"))
                self.status_bar_widget.setText(f"File: {filepath} | Status: Showing Raw {ext.upper()} Content")

            except Exception as e:
                self.status_bar_widget.setText(f"[ERROR] Could not read file: {e}")
        else:
            self.status_bar_widget.setText(f"[ERROR] Unsupported file extension for rendering: {ext}")


    def _generate_html_from_devcar(self, filepath):
        """
        Lightweight wrapper that uses the standalone renderer module when
        try:
            return render_file(filepath)
        except Exception as e:
            import traceback, html as _html
            return f"<div style='padding:16px;background:#fff3f2;border:1px solid #fca5a5;border-radius:8px;'>" \
                   f"<h2 style='color:#991b1b;'>Render Error</h2><pre>{_html.escape(str(e))}\n{_html.escape(traceback.format_exc())}</pre></div>"
                body {{ font-family: 'Inter', sans-serif; }}
            </style>
            <script>
                // State management
                window.devcarBindings = {{}};
                window.bindingCallbacks = new Set();
                
                // Safe HTML escaping
                function escapeHtml(unsafe) {{
                    return unsafe
                        .replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;")
                        .replace(/"/g, "&quot;")
                        .replace(/'/g, "&#039;");
                }}
                
                // Reactive updates
                function updateBindings() {{
                    const spans = document.querySelectorAll('[data-bind]');
                    spans.forEach(span => {{
                        const key = span.getAttribute('data-bind').trim();
                        const value = window.devcarBindings[key];
                        const formatted = value !== undefined ? escapeHtml(String(value)) : '';
                        if (span.textContent !== formatted) {{
                            span.textContent = formatted;
                        }}
                    }});
                    
                    // Call any registered callbacks
                    window.bindingCallbacks.forEach(callback => callback(window.devcarBindings));
                }}
                
                // Register update callback
                function onBindingUpdate(callback) {{
                    window.bindingCallbacks.add(callback);
                }}
                
                // Debounced updates
                let updateTimer;
                function debouncedUpdate() {{
                    clearTimeout(updateTimer);
                    updateTimer = setTimeout(updateBindings, 50);
                }}
                
                // Initialize
                document.addEventListener('DOMContentLoaded', updateBindings);
            </script>
        </head>
        <body class="bg-gray-100 min-h-screen">
            <div id="app-container" class="w-full h-full {current_layout}">
                <h1 class="text-4xl font-extrabold text-gray-900 mb-6 border-b pb-2">DevCAR Application Preview</h1>
                {html_body_content}
            </div>
        </body>
        </html>
        '''
    # removed stray return final_html (was unreachable)


# --- Low-Code Studio Classes ---

# --- Error Handling ---

class ErrorBus:
    """Central error handling and logging system"""
    def __init__(self):
        self.error_log = os.path.join('logs', 'studio.log')
        os.makedirs('logs', exist_ok=True)
        
    def capture(self, error: Exception, context: dict):
        """Capture and log an error with context"""
        timestamp = datetime.now().isoformat()
        error_id = str(uuid.uuid4())
        
        # Log to file
        try:
            with open(self.error_log, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] Error ID: {error_id}\n")
                f.write(f"Context: {json.dumps(context, ensure_ascii=False)}\n")
                f.write(f"Error: {str(error)}\n")
                f.write(f"Stack Trace:\n{traceback.format_exc()}\n\n")
        except Exception:
            # Best-effort logging: try a fallback write in binary to avoid crashes
            try:
                with open(self.error_log, 'ab') as fb:
                    fb.write(f"[{timestamp}] Error ID: {error_id}\n".encode('utf-8', errors='replace'))
                    fb.write(f"Error: {str(error)}\n".encode('utf-8', errors='replace'))
                    fb.write(f"Stack Trace:\n{traceback.format_exc()}\n\n".encode('utf-8', errors='replace'))
            except Exception:
                pass
            
        # Store in database
        if context.get('app_id'):
            with sqlite3.connect('studio.db') as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO errors (
                        id, timestamp, app_id, page_id, region_id, 
                        action_id, sql_hash, error_type, message, stack_trace
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    error_id, timestamp, context.get('app_id'),
                    context.get('page_id'), context.get('region_id'),
                    context.get('action_id'), context.get('sql_hash'),
                    type(error).__name__, str(error), traceback.format_exc()
                ))
                
        return error_id
        
    def show_error(self, parent: Any, error: Exception, context: dict):
        """Show user-friendly error dialog with details"""
        error_id = self.capture(error, context)
        
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Error")
        msg.setText(f"An error occurred: {str(error)}")
        
        details = f'''Error ID: {error_id}
Context: {json.dumps(context, indent=2)}
Time: {datetime.now().isoformat()}

Stack Trace:
{traceback.format_exc()}'''
        
        msg.setDetailedText(details)
        
        # Add "View Log" button
        view_log = msg.addButton("View Log", QMessageBox.ButtonRole.ActionRole)
        view_log.clicked.connect(lambda: self._show_log_viewer(parent))
        
        msg.exec()
        
    def _show_log_viewer(self, parent: Any):
        """Show log viewer dialog"""
        dialog = QDialog(parent)
        dialog.setWindowTitle("Error Log Viewer")
        dialog.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Log text view
        log_view = QTextEdit()
        log_view.setReadOnly(True)
        log_view.setFont(QFont('Courier New', 10))
        
        try:
            with open(self.error_log, 'r', encoding='utf-8', errors='replace') as f:
                log_view.setText(f.read())
        except Exception as e:
            log_view.setText(f"Error reading log: {e}")
            
        layout.addWidget(log_view)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec()

class DevCarStudioIDE(QMainWindow):
    """The low-code studio IDE interface"""
    def __init__(self):
        super().__init__()
        self.settings = get_app_settings()
        self.setWindowTitle(f"{self.settings['title']} - Studio IDE")
        self.setGeometry(100, 100, 1600, 900)
        
        # Initialize stores
        self.metadata_store = None  # Will be initialized with workspace
        self._init_workspace()
        
        # Track current application/page
        self.current_app_id = None
        self.current_page_id = None
        self.current_app_mode = 'browser'  # or 'windows'
        self.error_bus = ErrorBus()
        
        # Initialize UI
        self._init_ui()
        self._setup_drag_drop()
        self._load_themes()
        self._connect_signals()
        self._load_or_create_app()
        
    def _init_workspace(self):
        """Initialize or load workspace metadata"""
        workspace_path = os.path.join(os.getcwd(), 'devcar_workspace')
        os.makedirs(workspace_path, exist_ok=True)
        self.metadata_store = MetadataStore(workspace_path)
        
    def _load_or_create_app(self):
        """Load existing app or create new one"""
        # Check for existing apps
        apps = self.metadata_store.get_applications()
        
        if not apps:
            # Create default application if none exists
            name, ok = QInputDialog.getText(self, 
                "New Application", 
                "No applications found. Enter name for new application:")
            
            if ok and name:
                self.current_app_id = self.metadata_store.create_application(
                    name=name,
                    mode=self.current_app_mode
                )
                self.current_app_mode = 'browser'  # default mode
                self._update_explorer_tree()
            else:
                QMessageBox.critical(self, "Error", "Application required to continue")
                sys.exit(1)
        else:
            # Use first application as default
            self.current_app_id = apps[0]['id']
            self.current_app_mode = apps[0]['mode']
            self._update_explorer_tree()
            
    def _update_explorer_tree(self):
        """Update the explorer tree with current application structure"""
        if not hasattr(self, 'explorer_tree'):
            return
            
        self.explorer_tree.clear()
        
        if not self.current_app_id:
            return
            
        # Add application node
        app = self.metadata_store.get_application(self.current_app_id)
        app_node = QTreeWidgetItem(self.explorer_tree)
        app_node.setText(0, f"{app['name']} ({app['mode']})")
        app_node.setData(0, Qt.ItemDataRole.UserRole, {'type': 'app', 'id': app['id']})
        
        # Add pages
        pages = self.metadata_store.get_pages(self.current_app_id)
        for page in pages:
            page_node = QTreeWidgetItem(app_node)
            page_node.setText(0, page['name'])
            page_node.setData(0, Qt.ItemDataRole.UserRole, 
                            {'type': 'page', 'id': page['id']})
            
            # Add regions
            regions = self.metadata_store.get_regions(page['id'])
            for region in regions:
                region_node = QTreeWidgetItem(page_node)
                region_node.setText(0, f"{region['type']}: {region['title']}")
                region_node.setData(0, Qt.ItemDataRole.UserRole,
                                  {'type': 'region', 'id': region['id']})
                
                # Add items for form/IG regions
                if region['type'] in ('form', 'ig'):
                    items = self.metadata_store.get_items(region['id'])
                    for item in items:
                        item_node = QTreeWidgetItem(region_node)
                        item_node.setText(0, f"P{page['id']}_{item['name']}")
                        item_node.setData(0, Qt.ItemDataRole.UserRole,
                                        {'type': 'item', 'id': item['id']})
                                        
            # Add dynamic actions
            da_node = QTreeWidgetItem(page_node)
            da_node.setText(0, "Dynamic Actions")
            actions = self.metadata_store.get_dynamic_actions(page['id'])
            for action in actions:
                action_node = QTreeWidgetItem(da_node)
                action_node.setText(0, action['name'])
                action_node.setData(0, Qt.ItemDataRole.UserRole,
                                  {'type': 'action', 'id': action['id']})
                                  
            # Add processes
            proc_node = QTreeWidgetItem(page_node)
            proc_node.setText(0, "Processes")
            processes = self.metadata_store.get_processes(page['id'])
            for proc in processes:
                proc_node = QTreeWidgetItem(proc_node)
                proc_node.setText(0, f"{proc['point']}: {proc['name']}")
                proc_node.setData(0, Qt.ItemDataRole.UserRole,
                                {'type': 'process', 'id': proc['id']})
        
    def _init_ui(self):
        """Set up the main interface components"""
        self.setStyleSheet('''
            QMainWindow {
                background-color: #1E1E1E;
                color: #D4D4D4;
            }
            QTabWidget::pane {
                border: 1px solid #3C3C3C;
                background: #252526;
            }
            QTabBar::tab {
                background: #2D2D2D;
                color: #D4D4D4;
                padding: 8px 15px;
                border: 1px solid #3C3C3C;
            }
            QTabBar::tab:selected {
                background: #1E1E1E;
                border-bottom: none;
            }
            QDockWidget {
                border: 1px solid #3C3C3C;
            }
            QDockWidget::title {
                background: #2D2D2D;
                padding: 6px;
            }
            QTreeWidget {
                background-color: #252526;
                color: #D4D4D4;
                border: none;
            }
            QTreeWidget::item:hover {
                background-color: #2D2D2D;
            }
            QTreeWidget::item:selected {
                background-color: #094771;
            }
        ''')
        
        # Main layout with tab widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create main tab widget
        self.main_tabs = QTabWidget()
        main_layout.addWidget(self.main_tabs)
        
        # Add Code tab (original IDE view)
        code_tab = QWidget()
        code_layout = QHBoxLayout(code_tab)
        code_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add original IDE components to Code tab
        self.component_palette = self._create_component_palette()
        self.canvas = self._create_design_canvas()
        self.properties_panel = self._create_properties_panel()
        
        code_layout.addWidget(self.component_palette, stretch=1)
        code_layout.addWidget(self.canvas, stretch=2)
        code_layout.addWidget(self.properties_panel, stretch=1)
        
        self.main_tabs.addTab(code_tab, "Code")
        
        # Add Low-Code tab with 4 panes
        lowcode_tab = QWidget()
        lowcode_layout = QHBoxLayout(lowcode_tab)
        lowcode_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create left splitter for explorer and designer
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 1. Explorer pane
        explorer_widget = QWidget()
        explorer_layout = QVBoxLayout(explorer_widget)
        explorer_layout.setContentsMargins(0, 0, 0, 0)

        self.explorer_tree = QTreeWidget()
        self.explorer_tree.setHeaderLabel("Applications")
        explorer_layout.addWidget(self.explorer_tree)

        # Context menu for explorer: right-click to create pages/components
        self.explorer_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        def _on_explorer_context(point):
            item = self.explorer_tree.itemAt(point)
            menu = QMenu()
            new_page_action = QAction("New Page", self)
            def _create_page():
                # Create a simple page and refresh
                if not self.current_app_id:
                    return
                page_name, ok = QInputDialog.getText(self, "New Page", "Enter page name:")
                if ok and page_name:
                    self.metadata_store.create_page(self.current_app_id, page_name)
                    self._update_explorer_tree()
            new_page_action.triggered.connect(_create_page)
            menu.addAction(new_page_action)
            menu.exec(self.explorer_tree.viewport().mapToGlobal(point))
        self.explorer_tree.customContextMenuRequested.connect(_on_explorer_context)

        # Explorer click handler
        def _on_explorer_item_clicked_local(item: Any, column: int):
            try:
                data = item.data(0, Qt.ItemDataRole.UserRole)
            except Exception:
                data = None

            if not data:
                # error_console may not yet be created; use status bar as fallback
                try:
                    self.error_console.append("Explorer: clicked item has no data")
                except Exception:
                    pass
                return

            item_type = data.get('type')
            item_id = data.get('id')

            if item_type == 'page':
                try:
                    page_meta = self.metadata_store.get_page(item_id)
                    renderer = RuntimeRenderer(page_meta.get('mode', 'browser'))
                    html_or_widget = renderer.render_page(page_meta)
                    if isinstance(html_or_widget, str):
                        if hasattr(self, 'preview_view'):
                            self.preview_view.setHtml(html_or_widget, QUrl('file:///'))
                    else:
                        dlg = QDialog(self)
                        dlg.setWindowTitle(f"Preview: {page_meta.get('name')}")
                        lay = QVBoxLayout(dlg)
                        lay.addWidget(html_or_widget)
                        dlg.exec()
                    try:
                        self.error_console.append(f"Opened page: {page_meta.get('name')}")
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        self.error_console.append(f"Failed to open page: {e}")
                    except Exception:
                        pass
            else:
                try:
                    self.error_console.append(f"Explorer: clicked {item_type} (id={item_id})")
                except Exception:
                    pass

        # connect the local handler
        self.explorer_tree.itemClicked.connect(_on_explorer_item_clicked_local)

        left_splitter.addWidget(explorer_widget)
        
        # 2. Designer pane (Property grid)
        self.designer_widget = QWidget()
        designer_layout = QVBoxLayout(self.designer_widget)
        designer_layout.setContentsMargins(0, 0, 0, 0)
        
        self.designer_form = QFormLayout()
        designer_container = QWidget()
        designer_container.setLayout(self.designer_form)
        
        designer_scroll = QScrollArea()
        designer_scroll.setWidget(designer_container)
        designer_scroll.setWidgetResizable(True)
        designer_layout.addWidget(designer_scroll)
        
        left_splitter.addWidget(self.designer_widget)
        
        # Create right splitter for bindings and test run
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 3. Bindings pane
        bindings_widget = QWidget()
        bindings_layout = QVBoxLayout(bindings_widget)
        bindings_layout.setContentsMargins(0, 0, 0, 0)
        
        bindings_label = QLabel("Page Items & Bindings")
        bindings_layout.addWidget(bindings_label)
        
        self.bindings_table = QTableWidget()
        self.bindings_table.setColumnCount(3)
        self.bindings_table.setHorizontalHeaderLabels(["Item", "Value", "Bind Variable"])
        self.bindings_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        bindings_layout.addWidget(self.bindings_table)
        
        right_splitter.addWidget(bindings_widget)
        
        # 4. Test Run pane
        test_widget = QWidget()
        test_layout = QVBoxLayout(test_widget)
        test_layout.setContentsMargins(0, 0, 0, 0)
        
        # Test run controls
        test_controls = QHBoxLayout()
        self.test_mode = QComboBox()
        self.test_mode.addItems(['browser', 'windows'])
        test_controls.addWidget(QLabel("Mode:"))
        test_controls.addWidget(self.test_mode)
        
        run_button = QPushButton("Run Test")
        run_button.clicked.connect(self._run_test)
        test_controls.addWidget(run_button)
        
        test_layout.addLayout(test_controls)
        
        # Test output tabs
        test_tabs = QTabWidget()
        
        # Preview tab
        self.preview_view = QWebEngineView()
        test_tabs.addTab(self.preview_view, "Preview")
        
        # Error console tab
        self.error_console = QTextEdit()
        self.error_console.setReadOnly(True)
        test_tabs.addTab(self.error_console, "Error Console")
        
        # Event trace tab
        self.event_trace = QTextEdit()
        self.event_trace.setReadOnly(True)
        test_tabs.addTab(self.event_trace, "Event Trace")
        
        test_layout.addWidget(test_tabs)
        right_splitter.addWidget(test_widget)
        
        # Add splitters to layout
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 1)
        
        lowcode_layout.addWidget(main_splitter)
        self.main_tabs.addTab(lowcode_tab, "Low-Code Studio")
        
        # Create the toolbar and menu
        self._create_toolbar()
        self._create_menu()

    def _run_test(self):
        """Run the current page in the preview pane using the selected mode."""
        try:
            mode = self.test_mode.currentText() if hasattr(self, 'test_mode') else 'browser'

            # Prefer current_page_id; otherwise pick first page of current app
            page_id = getattr(self, 'current_page_id', None)
            if not page_id and getattr(self, 'current_app_id', None):
                pages = self.metadata_store.get_pages(self.current_app_id)
                page_id = pages[0]['id'] if pages else None

            if not page_id:
                if hasattr(self, 'error_console'):
                    self.error_console.append("Run Test: No page selected or available to run.")
                return

            page_meta = self.metadata_store.get_page(page_id)
            renderer = RuntimeRenderer(mode)
            html_or_widget = renderer.render_page(page_meta)

            if isinstance(html_or_widget, str):
                if hasattr(self, 'preview_view'):
                    self.preview_view.setHtml(html_or_widget, QUrl('file:///'))
            else:
                dlg = QDialog(self)
                dlg.setWindowTitle(f"Preview: {page_meta.get('name')}")
                lay = QVBoxLayout(dlg)
                lay.addWidget(html_or_widget)
                dlg.exec()

            if hasattr(self, 'error_console'):
                self.error_console.append(f"Run Test: Rendered page '{page_meta.get('name')}' in {mode} mode.")

        except Exception as e:
            if hasattr(self, 'error_console'):
                self.error_console.append(f"Run Test failed: {e}\n{traceback.format_exc()}")
        
    def _create_component_palette(self) -> Any:
        """Create the draggable component palette"""
        palette = QToolBox()
        
        # Layout components
        layout_group = QWidget()
        layout_list = QListWidget()
        layout_list.addItems(['Grid Layout', 'Flex Layout', 'Card Layout'])
        layout_group_layout = QVBoxLayout(layout_group)
        layout_group_layout.addWidget(layout_list)
        palette.addItem(layout_group, "Layouts")
        
        # Input components
        input_group = QWidget()
        input_list = QListWidget()
        input_list.addItems(['Text Input', 'Number Input', 'Date Input', 
                           'Checkbox', 'Radio Group', 'Select', 'Text Area'])
        input_group_layout = QVBoxLayout(input_group)
        input_group_layout.addWidget(input_list)
        palette.addItem(input_group, "Inputs")
        
        # Container components
        container_group = QWidget()
        container_list = QListWidget()
        container_list.addItems(['Card', 'Panel', 'Tabs', 'Accordion'])
        container_group_layout = QVBoxLayout(container_group)
        container_group_layout.addWidget(container_list)
        palette.addItem(container_group, "Containers")
        
        # Data components
        data_group = QWidget()
        data_list = QListWidget()
        data_list.addItems(['Table', 'Data Grid', 'List View', 'Tree View'])
        data_group_layout = QVBoxLayout(data_group)
        data_group_layout.addWidget(data_list)
        palette.addItem(data_group, "Data")
        
        # Chart components
        chart_group = QWidget()
        chart_list = QListWidget()
        chart_list.addItems(['Bar Chart', 'Line Chart', 'Pie Chart', 'Area Chart'])
        chart_group_layout = QVBoxLayout(chart_group)
        chart_group_layout.addWidget(chart_list)
        palette.addItem(chart_group, "Charts")
        
        # Navigation components
        nav_group = QWidget()
        nav_list = QListWidget()
        nav_list.addItems(['Menu Bar', 'Sidebar', 'Breadcrumb', 'Tabs'])
        nav_group_layout = QVBoxLayout(nav_group)
        nav_group_layout.addWidget(nav_list)
        palette.addItem(nav_group, "Navigation")
        
        return palette
        
    def _create_design_canvas(self) -> Any:
        """Create the main design surface"""
        canvas = QWidget()
        canvas.setAcceptDrops(True)
        canvas.setStyleSheet('''
            QWidget {
                background-color: #2D2D2D;
                border: 2px dashed #666666;
                border-radius: 5px;
            }
        ''')
        
        # Grid layout for snap-to-grid
        canvas_layout = QGridLayout(canvas)
        canvas_layout.setSpacing(10)
        canvas_layout.setContentsMargins(20, 20, 20, 20)
        
        return canvas
        
    def _create_properties_panel(self) -> Any:
        """Create the properties/inspector panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Component properties
        props_group = QGroupBox("Properties")
        props_layout = QFormLayout(props_group)
        
        self.prop_name = QLineEdit()
        props_layout.addRow("Name:", self.prop_name)
        
        self.prop_type = QComboBox()
        self.prop_type.addItems(['text', 'number', 'date', 'boolean'])
        props_layout.addRow("Type:", self.prop_type)
        
        self.prop_required = QCheckBox("Required")
        props_layout.addRow("", self.prop_required)
        
        layout.addWidget(props_group)
        
        # Data binding
        data_group = QGroupBox("Data Binding")
        data_layout = QFormLayout(data_group)
        
        self.data_source = QComboBox()
        self.data_source.addItems(['None', 'REST API', 'GraphQL', 'Database'])
        data_layout.addRow("Source:", self.data_source)
        
        self.data_path = QLineEdit()
        data_layout.addRow("Path:", self.data_path)
        
        layout.addWidget(data_group)
        
        # Events/Actions
        events_group = QGroupBox("Events")
        events_layout = QFormLayout(events_group)
        
        self.event_type = QComboBox()
        self.event_type.addItems(['onClick', 'onChange', 'onSubmit'])
        events_layout.addRow("Event:", self.event_type)
        
        self.event_action = QComboBox()
        self.event_action.addItems(['Navigate To', 'Submit Form', 'Custom Action'])
        events_layout.addRow("Action:", self.event_action)
        
        layout.addWidget(events_group)
        
        # Style properties
        style_group = QGroupBox("Style")
        style_layout = QFormLayout(style_group)
        
        self.style_theme = QComboBox()
        self.style_theme.addItems(['Default', 'Primary', 'Success', 'Danger'])
        style_layout.addRow("Theme:", self.style_theme)
        
        self.style_size = QComboBox()
        self.style_size.addItems(['Small', 'Medium', 'Large'])
        style_layout.addRow("Size:", self.style_size)
        
        layout.addWidget(style_group)
        
        # Stretch to fill space
        layout.addStretch()
        
        return panel
        
    def _create_toolbar(self):
        """Create the main toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # New page
        new_action = QAction(QIcon.fromTheme('document-new'), 'New Page', self)
        new_action.setStatusTip('Create a new page')
        new_action.triggered.connect(self.new_page)
        toolbar.addAction(new_action)
        
        # Save
        save_action = QAction(QIcon.fromTheme('document-save'), 'Save', self)
        save_action.setStatusTip('Save the current page')
        save_action.triggered.connect(self.save_page)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # Preview
        preview_action = QAction(QIcon.fromTheme('view-preview'), 'Preview', self)
        preview_action.setStatusTip('Preview the current page')
        preview_action.triggered.connect(self.preview_page)
        toolbar.addAction(preview_action)
        
        # Generate code
        generate_action = QAction(QIcon.fromTheme('applications-development'), 'Generate Code', self)
        generate_action.setStatusTip('Generate DevCAR code')
        generate_action.triggered.connect(self.generate_code)
        toolbar.addAction(generate_action)
        
    def _create_menu(self):
        """Create the main menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        new_page = QAction('New Page', self)
        new_page.setShortcut('Ctrl+N')
        new_page.triggered.connect(self.new_page)
        file_menu.addAction(new_page)
        
        open_page = QAction('Open Page', self)
        open_page.setShortcut('Ctrl+O')
        open_page.triggered.connect(self.open_page)
        file_menu.addAction(open_page)
        
        save_page = QAction('Save Page', self)
        save_page.setShortcut('Ctrl+S')
        save_page.triggered.connect(self.save_page)
        file_menu.addAction(save_page)
        
        # Edit menu
        edit_menu = menubar.addMenu('&Edit')
        
        undo = QAction('Undo', self)
        undo.setShortcut('Ctrl+Z')
        edit_menu.addAction(undo)
        
        redo = QAction('Redo', self)
        redo.setShortcut('Ctrl+Y')
        edit_menu.addAction(redo)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        preview = QAction('Preview', self)
        preview.setShortcut('F5')
        preview.triggered.connect(self.preview_page)
        view_menu.addAction(preview)
        
        # Tools menu
        tools_menu = menubar.addMenu('&Tools')
        
        generate = QAction('Generate Code', self)
        generate.triggered.connect(self.generate_code)
        tools_menu.addAction(generate)
        
        deploy = QAction('Deploy Application', self)
        deploy.triggered.connect(self.deploy_application)
        tools_menu.addAction(deploy)
        
    def _setup_drag_drop(self):
        """Configure drag and drop for components"""
        for i in range(self.component_palette.count()):
            page = self.component_palette.widget(i)
            if isinstance(page, QWidget):
                for child in page.findChildren(QListWidget):
                    child.setDragEnabled(True)
                    child.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        
        self.canvas.setAcceptDrops(True)
        
    def _load_themes(self):
        """Load available themes"""
        self.themes = {
            'default': {
                'primary': '#3B82F6',
                'secondary': '#6B7280',
                'success': '#10B981',
                'danger': '#EF4444',
                'warning': '#F59E0B',
                'info': '#3B82F6'
            }
        }
        
    def _connect_signals(self):
        """Connect various UI signals to handlers"""
        # Property changes
        self.prop_name.textChanged.connect(self._on_property_changed)
        self.prop_type.currentTextChanged.connect(self._on_property_changed)
        self.prop_required.stateChanged.connect(self._on_property_changed)
        
        # Data binding changes
        self.data_source.currentTextChanged.connect(self._on_data_binding_changed)
        self.data_path.textChanged.connect(self._on_data_binding_changed)
        
        # Style changes
        self.style_theme.currentTextChanged.connect(self._on_style_changed)
        self.style_size.currentTextChanged.connect(self._on_style_changed)
        
    def new_page(self):
        """Create a new page in the current application"""
        app_id = self.current_app_id  # Would be set when opening an application
        if not app_id:
            QMessageBox.warning(self, "Warning", "Please open an application first")
            return
            
        name, ok = QInputDialog.getText(self, "New Page", "Enter page name:")
        if ok and name:
            page_id = self.metadata_store.create_page(app_id, name)
            self._load_page(page_id)
            
    def open_page(self):
        """Open an existing page"""
        # This would show a dialog with available pages
        pass
        
    def save_page(self):
        """Save the current page state"""
        if not hasattr(self, 'current_page_id'):
            QMessageBox.warning(self, "Warning", "No page is currently open")
            return
            
        # Collect current canvas state
        components = self._collect_canvas_state()
        
        # Update metadata
        self.metadata_store.update_page(self.current_page_id, components)
        
        QMessageBox.information(self, "Success", "Page saved successfully")
        
    def preview_page(self):
        """Preview the current page design"""
        if not hasattr(self, 'current_page_id'):
            QMessageBox.warning(self, "Warning", "No page is currently open")
            return
            
        # Generate preview HTML
        preview = self.generate_preview()
        
        # Show in web view
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("Page Preview")
        preview_dialog.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout(preview_dialog)
        web_view = QWebEngineView()
        web_view.setHtml(preview)
        layout.addWidget(web_view)
        
        preview_dialog.exec()
        
    def generate_code(self):
        """Generate DevCAR code for the current page"""
        if not hasattr(self, 'current_page_id'):
            QMessageBox.warning(self, "Warning", "No page is currently open")
            return
            
        # Generate code
        code = self._generate_devcar_code()
        
        # Show code dialog
        code_dialog = QDialog(self)
        code_dialog.setWindowTitle("Generated Code")
        code_dialog.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout(code_dialog)
        
        # Code editor
        code_edit = QTextEdit()
        code_edit.setFont(QFont('Courier New', 10))
        code_edit.setPlainText(code)
        layout.addWidget(code_edit)
        
        # Buttons
        button_layout = QHBoxLayout()
        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(code))
        button_layout.addWidget(copy_button)
        
        save_button = QPushButton("Save As...")
        save_button.clicked.connect(lambda: self._save_generated_code(code))
        button_layout.addWidget(save_button)
        
        layout.addLayout(button_layout)
        code_dialog.exec()
        
    def deploy_application(self):
        """Deploy the current application"""
        # This would handle packaging and deployment
        pass
        
    def _collect_canvas_state(self) -> List[dict]:
        """Collect the current state of components on the canvas"""
        components = []
        
        # Traverse the canvas grid layout
        grid = self.canvas.layout()
        for i in range(grid.count()):
            widget = grid.itemAt(i).widget()
            if widget:
                components.append({
                    'type': widget.component_type,
                    'properties': widget.properties,
                    'position': {
                        'row': grid.getItemPosition(i)[0],
                        'column': grid.getItemPosition(i)[1]
                    }
                })
                
        return components

    def _load_page(self, page_id: str):
        """Load a page by id into the canvas and preview"""
        try:
            page_meta = self.metadata_store.get_page(page_id)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load page: {e}")
            return

        # Set the current page id
        self.current_page_id = page_id

        # Clear canvas
        try:
            layout = self.canvas.layout()
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        except Exception:
            pass

        # Add components if any
        components = page_meta.get('components') or page_meta.get('regions') or []
        for comp in components:
            try:
                # Try to use previously implemented factory
                self._create_component(comp)
            except Exception:
                # ignore if unknown component
                pass

        # Apply bindings if present
        bindings = page_meta.get('bindings', [])
        try:
            self._apply_bindings(bindings)
        except Exception:
            pass

        # Update explorer tree selection
        try:
            self._update_explorer_tree()
        except Exception:
            pass

        # Render preview if preview exists
        try:
            renderer = RuntimeRenderer(page_meta.get('mode', 'browser'))
            html = renderer.render_page(page_meta)
            if isinstance(html, str) and hasattr(self, 'preview_view'):
                self.preview_view.setHtml(html, QUrl('file:///'))
        except Exception:
            pass
        
    def _generate_preview(self) -> str:
        """Generate preview HTML for the current page"""
        components = self._collect_canvas_state()
        context = {'user': {'name': 'Preview User'}}
        
        generator = HTMLGenerator(context)
        return generator.generate(self._components_to_ast(components))
        
    def _generate_devcar_code(self) -> str:
        """Generate DevCAR code for the current page"""
        components = self._collect_canvas_state()
        
        code = []
        code.append("# Generated by DevCAR Studio")
        code.append(f"# Page: {self.current_page_id}")
        code.append("")
        
        # Add imports if needed
        code.append("# Layout")
        layout_component = next((c for c in components if c['type'].endswith('Layout')), None)
        if layout_component:
            code.append(f"LAYOUT[{layout_component['properties']['type']}, columns={layout_component['properties'].get('columns', 1)}]")
            code.append("")
            
        # Add components
        for component in components:
            if component['type'] == 'Card':
                code.extend(self._generate_card_code(component))
            elif component['type'].endswith('Input'):
                code.extend(self._generate_input_code(component))
            elif component['type'] == 'Table':
                code.extend(self._generate_table_code(component))
            elif component['type'].endswith('Chart'):
                code.extend(self._generate_chart_code(component))
                
        return '\n'.join(code)
        
    def _generate_card_code(self, component: dict) -> List[str]:
        """Generate code for a card component"""
        props = component['properties']
        code = []
        
        code.append(f"COMPONENT[CARD, Title={props['title']}, class={props.get('class', '')}]")
        if 'content' in props:
            code.append(f"    {props['content']}")
        code.append("END_COMPONENT")
        
        return code
        
    def _generate_input_code(self, component: dict) -> List[str]:
        """Generate code for an input component"""
        props = component['properties']
        type_map = {
            'text': 'text',
            'number': 'number',
            'date': 'date',
            'boolean': 'checkbox'
        }
        
        input_type = type_map.get(props['type'], 'text')
        code = []
        
        if input_type == 'checkbox':
            code.append(f"CHECKBOX[{props['name']}, default={props.get('default', 'false')}]")
        else:
            code.append(f"INPUT[{props['name']}, type={input_type}, placeholder={props.get('placeholder', '')}]")
            
        return code
        
    def _generate_table_code(self, component: dict) -> List[str]:
        """Generate code for a table component"""
        props = component['properties']
        columns = ', '.join(props['columns'])
        
        return [f"CREATE_TABLE[{props['title']}, {columns}]"]
        
    def _generate_chart_code(self, component: dict) -> List[str]:
        """Generate code for a chart component"""
        props = component['properties']
        data = ' '.join(map(str, props['data']))
        
        return [f"CHART[{props['title']}, type={props['type']}, data={data}]"]
        
    def _save_generated_code(self, code: str):
        """Save the generated code to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Code", "", "DevCAR Files (*.dev);;All Files (*.*)")
            
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                QMessageBox.information(self, "Success", "Code saved successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save code: {e}")
                
    def _on_property_changed(self):
        """Handle property changes in the properties panel"""
        if not hasattr(self, 'selected_component'):
            return
            
        self.selected_component.properties.update({
            'name': self.prop_name.text(),
            'type': self.prop_type.currentText(),
            'required': self.prop_required.isChecked()
        })
        
    def _on_data_binding_changed(self):
        """Handle data binding changes"""
        if not hasattr(self, 'selected_component'):
            return
            
        self.selected_component.properties.update({
            'data_source': self.data_source.currentText(),
            'data_path': self.data_path.text()
        })
        
    def _on_style_changed(self):
        """Handle style property changes"""
        if not hasattr(self, 'selected_component'):
            return
            
        self.selected_component.properties.update({
            'theme': self.style_theme.currentText(),
            'size': self.style_size.currentText()
        })
        self._update_component_style(self.selected_component)
        
    def _update_component_style(self, component):
        """Update the visual style of a component"""
        theme = self.themes['default']
        component_theme = component.properties.get('theme', 'default').lower()
        
        if component_theme in theme:
            color = theme[component_theme]
            component.setStyleSheet(f'''
                QWidget {{
                    background-color: {color}15;
                    border: 2px solid {color};
                    border-radius: 4px;
                    padding: 8px;
                }}
            ''')

# --- MAIN ENTRY POINT ---

def main():
    # If Qt is not available, we cannot run the GUI. Print an error and exit.
    if not _QT_AVAILABLE:
        print('''
        [ERROR] GUI framework not found.
        This application requires either PyQt6 or PySide6 to run.
        Please install one of them, for example:
            pip install PyQt6
        ''')
        # Fallback to HTML rendering if a .dev file is provided
        filepath_to_open = next((arg for arg in sys.argv[1:] if not arg.startswith('--')), None)
        if filepath_to_open and os.path.splitext(filepath_to_open)[1].lower() in DEV_EXTENSIONS:
            print(f"[INFO] Attempting to fall back to HTML rendering for {filepath_to_open}...")
            try:
                with open(filepath_to_open, 'r', encoding='utf-8') as f:
                    source = f.read()
                render_devcar_html(source)
            except Exception as e:
                print(f"[ERROR] Could not render HTML: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)

    # Handle command line args
    filepath_to_open = None
    studio_mode = False
    
    for arg in sys.argv[1:]:
        if arg == '--studio':
            studio_mode = True
        elif not arg.startswith('--'):
            filepath_to_open = arg

    # Determine which mode to launch
    if studio_mode:
        # Studio Mode: Launch the low-code IDE
        main_window = DevCarStudioIDE()
    elif filepath_to_open and os.path.splitext(filepath_to_open)[1].lower() in DEV_EXTENSIONS:
        # Launcher Mode: If a DevCAR file is opened, launch the rendered app
        main_window = DevCarLauncher(filepath_to_open)
    elif filepath_to_open and os.path.exists(filepath_to_open):
        # IDE Mode: If any other file exists, open it in the IDE
        main_window = DevCarIDE(filepath_to_open)
    else:
        # Show mode selection dialog
        mode = QMessageBox.question(None, "Select Mode",
                                  "Choose IDE Mode:\n\nYes: Low-Code Studio\nNo: Code Editor",
                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if mode == QMessageBox.StandardButton.Yes:
            main_window = DevCarStudioIDE()
        else:
            main_window = DevCarIDE()

    main_window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
'''