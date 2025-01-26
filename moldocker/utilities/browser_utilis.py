import os, time
import copy, json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from scipy.stats import t
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QToolButton,
                               QDialog, QPushButton, QProgressBar, QLineEdit,
                               QTabWidget)
from PySide6.QtCore import QObject, Signal, Slot, QUrl, QJsonValue
from PySide6.QtGui import QColor, Qt, QAction, QGuiApplication
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import (QWebEnginePage, QWebEngineUrlRequestInterceptor,
                                     QWebEngineProfile, QWebEngineSettings)

from .adblock_parser import AdblockRules
from .ligplot_utilis import create_ligplot_figure

rules_pth = os.path.join(os.path.dirname(__file__), 'adblock_rules', 'rules.txt')
cookies_pth = os.path.join(os.path.dirname(__file__), 'cookies')
global rules
rules = None

if not os.path.isdir(cookies_pth):
    os.makedirs(cookies_pth)

class WebChannelHandler(QObject):
    coordinatesClicked = Signal(dict)
    contactReceived = Signal(dict, str)
    
    @Slot(QJsonValue)
    def receiveCoordinates(self, message: QJsonValue):
        coordinates = message.toObject()
        self.coordinatesClicked.emit(coordinates)
        
    @Slot(QJsonValue)
    def receiveContacts(self, message: QJsonValue):
        contacts_name = json.loads(message.toString())
        self.contactReceived.emit(contacts_name[0], contacts_name[1])

class ProteinCoordinateSignals(QObject):
    send_coordinates = Signal(dict)
    
class ProteinLigandEmbedSignals(QObject):
    contactDone = Signal(str)

class JSLogWebEnginePage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print(f"JS Console: {message} (Source: {sourceID}, Line: {lineNumber})")
        
class JSLogWebEnginePageSuppress(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        pass

class ProteinCoordinateBrowserWidget(QWidget):
    def __init__(self, mode: str):
        super().__init__()
        curr_dir = os.path.dirname(__file__)
        html_pth = os.path.join(curr_dir, 'ngl_web', 'webapp_position.html')
        self.signals = ProteinCoordinateSignals()
        
        self.browser = QWebEngineView()
        # self.browser.setPage(JSLogWebEnginePage(self))
        self.browser.setUrl(QUrl.fromLocalFile(html_pth))
        self.browser.loadFinished.connect(lambda _, m=mode: self.setup_theme(m))
        
        self.channel = QWebChannel()
        self.handler = WebChannelHandler()
        self.handler.coordinatesClicked.connect(self.get_coordinates)
        self.channel.registerObject("pywebchannel", self.handler)
        
        self.browser.page().setWebChannel(self.channel)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.browser)
        self.setLayout(layout)
        
        # self.setWindowTitle("Protein Structure Viewer")
    
    def setup_theme(self, mode: str):
        self.browser.page().runJavaScript(f"NGL.StageWidget(stage, '{mode}')")
        
    def load_protein_file(self, protein_pth: str):
        self.browser.page().runJavaScript(f"loadCustomFile('{protein_pth}');")
    
    def load_protein_string(self, pdbqt_string: str, scheme: str):
        self.browser.page().runJavaScript(f"loadCustomPDBQTString(`{pdbqt_string}`, `{scheme}`);")
        
    def clear_stage(self):
        self.browser.page().runJavaScript(f"clearAllStageComponents();")
        
    def create_center_sphere(self, vec, colors):
        self.browser.page().runJavaScript(f"createSphere({vec}, {colors}, 0.5);")
        
    def remove_sphere(self):
        self.browser.page().runJavaScript(f"rmSphere();")
        
    def create_bounding_box(self, center, colors, box, opacity):
        vec = f'[{center[0]}, {center[1]}, {center[2]}]'
        bb_pos = []   # x_min, x_max, y_min, y_max, z_min, z_max
        for cen, width in zip(center, box):
            half = float(width) / 2
            bb_pos.append(float(cen) - half)
            bb_pos.append(float(cen) + half)
        self.browser.page().runJavaScript(f"createBox({vec}, {colors}, {box[0]}, {box[1]}, {box[2]}, {opacity}, "
                                          f"{bb_pos[0]}, {bb_pos[1]}, {bb_pos[2]}, {bb_pos[3]}, {bb_pos[4]}, {bb_pos[5]});")
    
    def remove_box(self):
        self.browser.page().runJavaScript(f"rmBox();")
    
    def reorient_to_boundingbox(self):
        self.browser.page().runJavaScript(f"reorient();")
    
    def get_coordinates(self, coordinate_dict: dict):
        self.signals.send_coordinates.emit(coordinate_dict)
        
    def show_sidechains(self, position: str):
        self.browser.page().runJavaScript(f"showSideChain(`{position}`);")
        
    def set_box_visibility(self, checked: bool):
        self.browser.page().runJavaScript(f"setBoxVisibility(`{checked}`);")
        
    def set_center_visibility(self, checked: bool):
        self.browser.page().runJavaScript(f"setCenterVisibility(`{checked}`);")
    
    def set_highlight(self, hightlight_sel: str):
        self.browser.page().runJavaScript(f"updateHighlight(`{hightlight_sel}`);")

class ProteinLigandBrowserWidget(QWidget):
    def __init__(self):
        super().__init__()
        curr_dir = os.path.dirname(__file__)
        html_pth = os.path.join(curr_dir, 'ngl_web', 'webapp_display.html')
        
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl.fromLocalFile(html_pth))
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.browser)
        self.setLayout(layout)
        self.setWindowTitle('Protein Viewer')
        # screen = self.screen()
        # screen_size = screen.availableGeometry()
        # self.resize(screen_size.width() * 0.8, screen_size.height() * 0.6)
        
    def load_protein_file(self, protein_pth: str):
        self.browser.page().runJavaScript(f"loadCustomFile('{protein_pth}');")

class ProteinLigandEmbedBrowserWidget(QWidget):
    def __init__(self, theme: str):
        super().__init__()
        self.color_list = [
            "#1E90FF",  # Dodger Blue
            "#32CD32",  # Lime Green
            "#FFD700",  # Gold
            "#FF6347",  # Tomato
            "#9400D3",  # Dark Violet
            "#00CED1",  # Dark Turquoise
            "#FF4500",  # Orange Red
            "#8A2BE2",  # Blue Violet
            "#5F9EA0",  # Cadet Blue
            "#FF1493",  # Deep Pink
            "#4B0082",  # Indigo
            "#48D1CC",  # Medium Turquoise
            "#7CFC00",  # Lawn Green
            "#20B2AA",  # Light Sea Green
            "#FF69B4",  # Hot Pink
            "#ADFF2F",  # Green Yellow
            "#00FF7F",  # Spring Green
            "#FFA500",  # Orange
            "#00BFFF",  # Deep Sky Blue
            "#DA70D6",  # Orchid
            ]
        self.shown_sidechain_dict = {}
        self.shown_contact_dict = {}
        self.block_contact_dict = {}
        curr_dir = os.path.dirname(__file__)
        html_pth = os.path.join(curr_dir, 'ngl_web', 'webapp_embed.html')
        self.signals = ProteinLigandEmbedSignals()
        
        self.browser = QWebEngineView()
        # self.browser.setPage(JSLogWebEnginePage(self))
        self.browser.setUrl(QUrl.fromLocalFile(html_pth))
        self.browser.loadFinished.connect(lambda _, x=theme: self.setup_theme(x))
        
        self.channel = QWebChannel()
        self.handler = WebChannelHandler()
        self.handler.contactReceived.connect(self.process_contacts)
        self.channel.registerObject("pywebchannel", self.handler)
        self.browser.page().setWebChannel(self.channel)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.browser)
        self.setLayout(layout)
        self.color_cnt = 0
        
    def load_protein_file(self, protein_pth: str):
        self.browser.page().runJavaScript(f"loadCustomFile('{protein_pth}');")
            
    def load_protein_ligand_pdbqt_string(self, pdbqt_str: str, name: str, block_contact: bool):
        if name not in self.shown_sidechain_dict:
            self.shown_sidechain_dict[name] = ''
            self.block_contact_dict[name] = block_contact
            color = self.color_list[self.color_cnt % len(self.color_list)]
            self.color_cnt += 1
            self.browser.page().runJavaScript(f"loadProteinLigandPDBString(`{pdbqt_str}`, `{name}`, `{color}`);")
    
    def clear_stage(self):
        self.browser.page().runJavaScript(f"clearAllStageComponents();")
        self.shown_sidechain_dict = {}
        self.shown_contact_dict = {}
        
    def setup_theme(self, mode: str):
        self.browser.page().runJavaScript(f"setTheme('{mode}');")
        
    def reorient(self):
        self.browser.page().runJavaScript(f"reorient();")
        
    def remove_name_from_webpage(self, name: str):
        if name in self.shown_sidechain_dict:
            del self.shown_sidechain_dict[name]
            if name in self.shown_contact_dict:
                del self.shown_contact_dict[name]
            self.browser.page().runJavaScript(f"clearComponentByName(`{name}`);")
    
    def remove_all_name_from_webpage(self):
        for name in self.shown_sidechain_dict:
            self.browser.page().runJavaScript(f"clearComponentByName(`{name}`);")
        self.shown_sidechain_dict = {}
        self.shown_contact_dict = {}
        self.color_cnt = 0
    
    def process_contacts(self, contact_dict: dict, name: str):
        self.shown_contact_dict[name] = contact_dict
        if not self.block_contact_dict[name]:
            self.signals.contactDone.emit(name)
    
    def _get_aa_position(self, name_dict: dict):
        selection_name = name_dict['name']
        component_name = name_dict['modelName']
        if not selection_name.startswith('[UNL]'):
            res_chain = tuple(selection_name.split(']')[1].split('.')[0].split(':'))
            if res_chain not in self.shown_sidechain_dict[component_name]:
                self.shown_sidechain_dict[component_name].append(res_chain)
            else:
                self.shown_sidechain_dict[component_name].remove(res_chain)
            flex, cont = self.create_representation(self.shown_sidechain_dict[component_name])
            self.browser.page().runJavaScript(f"showSideChain(`{component_name}`, `{flex}`, `{cont}`);")
    
    def _create_representation(self, target_list: list):
        text_1_list = []
        text_2_list = []
        contact_list = []
        for res_chain in target_list:
            res, chain = res_chain
            text_1_list.append(f'{res}:{chain}.CA')
            text_2_list.append(f'{res}:{chain} and sidechain')
            contact_list.append(f'{res}:{chain}')
        if bool(text_1_list) & bool(text_2_list):
            return ' OR '.join(text_1_list) + ' OR (' + ' OR '.join(text_2_list) + ')', ' OR '.join(contact_list) + ' OR UNL'
        else:
            return 'none', 'none'

class FragmentPlotBrowserWidget(QWidget):
    def __init__(self, html_pth: str):
        super().__init__()
        
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl.fromLocalFile(html_pth))
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.browser)
        
        self.setLayout(layout)
        self.setWindowTitle('Energy vs. Energy Rank Plot')

class PlotViewer(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plotly_html_config = {'displaylogo': False,
                                   'toImageButtonOptions': {'format': 'png',
                                                            'scale' : 3},
                                   'edits': {'legendPosition'  : True,
                                             'legendText'      : True,
                                             'titleText'       : True,
                                             'colorbarPosition': True,},
                                   'showTips': False}
        self.black_color = QColor()
        self.black_color.setRgb(17, 17, 17)
    
    def setup_background_color(self, template: str):
        if '_dark' in template:
            self.page().setBackgroundColor(self.black_color)
        else:
            self.page().setBackgroundColor('white')
            
    def setup_html(self, html_f_pth: str):
        self.setUrl(QUrl.fromLocalFile(html_f_pth))
        
    def plot_histogram(self, df: pd.DataFrame, col: str, template_name: str, temp_pth: str):
        if col in df.columns:
            self.setup_background_color(template_name)
            rug_text = [df['Name'].to_list()]
            value = df[col].to_numpy()
            if col == 'Formal Charge':
                bin_size = 1
            else:
                iqr = np.percentile(value, 75) - np.percentile(value, 25)
                fd_bin_size = 2 * iqr / value.shape[0] ** (1 / 3)   # Freedman–Diaconis rule, 2 * IQR / n ^ (1/3)
                sturges_bin_size = (value.max() - value.min()) / np.ceil(np.log2(value.shape[0])) # Sturges
                bin_size = min(sturges_bin_size, fd_bin_size)   # the min between fd and sturges, same as numpy
            if col in ['Hydrogen Bond Donors', 'Hydrogen Bond Acceptors', 'Rotatable Bonds',
                       'Number of Rings', 'Number of Heavy Atoms', 'Number of Atoms']:
                bin_size = np.ceil(bin_size)
            bin_num = np.ceil((value.max() - value.min()) / bin_size)
            count, bin_edge = np.histogram(value, int(bin_num))
            bin_left, bin_right = bin_edge[:-1], bin_edge[1:]
            self.fig = ff.create_distplot([value], [col], bin_size=bin_size, rug_text=rug_text)
            histogram, kde, rug = self.fig['data']
            histogram.update({'name': f'Histogram', 'legendgroup': 'Histogram', 'histnorm': 'probability density',
                            'customdata': np.array([bin_left, bin_right, count]).T,
                            'hovertemplate': 'Range: %{customdata[0]:.4f}~%{customdata[1]:.4f}<br>Count: %{customdata[2]:d}<extra></extra>'})
            histogram['marker'].update({'line': {'color': 'white', 'width': 1}})
            kde.update({'name': 'KDE', 'legendgroup': 'KDE', 'showlegend': True, 'fill': 'tozeroy', 'fillcolor': 'rgba(250, 171, 92, 0.1)'})
            kde['marker'].update({'color': 'rgb(250, 171, 92)'})
            rug.update({'name': 'Rug', 'legendgroup': 'Rug', 'showlegend': True, 'marker_size': 25, 'hovertemplate': '<b>%{text}</b><br><br>'+col+': %{x}<extra></extra>'})
            rug['marker'].update({'color': 'rgb(35, 145, 88)'})
            self.fig['layout']['legend'].update({'traceorder': 'normal', 'yanchor': 'top', 'xanchor': 'right', 'y': 0.99, 'x': 0.99})
            self.fig['layout'].update({'title': f'{col} Distribution', 'margin': {'l': 10, 'r': 10, 't': 40, 'b': 25}})
            self.fig['layout'].update({'template': template_name, 'clickmode': 'select'})
            config = copy.deepcopy(self.plotly_html_config)
            config['toImageButtonOptions']['filename'] = f'{'_'.join(col.split(' '))}_Distribution'
            self.fig.write_html(temp_pth, config)
            return True
        return False
    
    def plot_margin(self, df: pd.DataFrame, x_col: str, y_col: str, color: str, template_name: str, temp_pth: str):
        if 'QED' in df.columns:
            self.setup_background_color(template_name)
            # self.fig = FigureResampler(px.scatter(df, x=x_col, y=y_col, marginal_x='histogram', marginal_y='histogram', hover_name='Name'))
            if not color:
                color = None
            self.fig = px.scatter(df, x=x_col, y=y_col, marginal_x='histogram', marginal_y='histogram', hover_name='Name', color=color)
            self.fig['data'][0].update({'opacity': 0.8})
            if template_name == 'presentation':
                l, r, t, b = 80, 50, 50, 50
            elif 'grido' in template_name or template_name == 'none':
                l, r, t, b = 60, 5, 40, 30
            else:
                l, r, t, b = 30, 5, 40, 30
            self.fig['layout'].update({'title': f'{x_col} V.S. {y_col}', 'margin': {'l': l, 'r': r, 't': t, 'b': b}})
            self.fig['layout'].update({'template': template_name, 'clickmode': 'select', 'modebar_add': ['togglespikelines']})
            config = copy.deepcopy(self.plotly_html_config)
            f_name = '_'.join(x_col.split(' '))+'_vs_'+'_'.join(y_col.split(' '))
            config['toImageButtonOptions']['filename'] = f'{f_name}_Marginal'
            self.fig.write_html(temp_pth, config)
            return True
        return False
        
    def calculate_pearson_corr(self, df: pd.DataFrame):
        total_list = []
        cols = df.columns
        for col in cols:
            total_list.append(np.array(df[col]))
        result = np.stack(total_list, axis=0)
        return np.corrcoef(result)
    
    def p_coef(self, df):
        np.seterr('ignore')
        total_list = []
        cols = df.columns
        for col in cols:
            total_list.append(np.array(df[col]))
        result = np.stack(total_list, axis=0)
        combine = np.vstack(result)
        vec_mean = np.mean(combine, axis=-1)
        vec_diff = combine - vec_mean[:, None]
        sum_square = np.sum(vec_diff ** 2, axis=-1) ** 0.5
        numerator = np.einsum('ij,kj->ik', vec_diff, vec_diff)
        denominator = np.outer(sum_square, sum_square)
        corr_mat = numerator / denominator
        t_mat = corr_mat * ((combine.shape[-1] - 2) / (1 - corr_mat ** 2)) ** 0.5
        p_val = 2 * (1 - t.cdf(np.abs(t_mat), combine.shape[-1] - 2))
        np.fill_diagonal(p_val, 1.)
        return corr_mat, p_val
    
    def plot_correlation(self, df: pd.DataFrame, template_name: str, temp_pth: str):
        if 'QED' in df.columns:
            self.setup_background_color(template_name)
            df = df.select_dtypes('number')
            cols = df.columns
            corr_mat, p_coef = self.p_coef(df)
            self.fig = px.imshow(corr_mat, x=cols, y=cols,
                                 color_continuous_scale='RdBu_r',
                                 range_color=[-1, 1])
            self.fig['data'][0].update({'hovertemplate': "X: %{x}<br>Y: %{y}<br>Pearson's r: %{customdata[0]:.4f}<br>P-value: %{customdata[1]:.4f}<extra></extra>",
                                        'customdata': np.stack([corr_mat, p_coef], axis=-1)})
            self.fig['layout'].update({'template': template_name})
            if template_name == 'presentation':
                self.fig['layout'].update({'title': 'Correlation Plot',
                                           'margin': {'l': 120, 'r': 10, 't': 50, 'b': 260},
                                           'clickmode': 'select'})
            elif 'grido' in template_name or template_name == 'none':
                self.fig['layout'].update({'title': 'Correlation Plot',
                                           'margin': {'l': 80, 'r': 80, 't': 80, 'b': 100},
                                           'clickmode': 'select'})
            else:
                self.fig['layout'].update({'title': 'Correlation Plot', 'clickmode': 'select'})
            config = copy.deepcopy(self.plotly_html_config)
            config['toImageButtonOptions']['filename'] = f'Correlation'
            self.fig.write_html(temp_pth, config)
            return True
        return False
        
    def plot_roc_curve(self, decision_dict: dict, column_name: str, template_name: str, temp_pth, applied_filter_dict=None):
        self.setup_background_color(template_name)
        self.fig = go.Figure()
        auc = decision_dict['AUC']
        thres = decision_dict['Thresholds']
        youden = decision_dict['Youden']
        customdata = np.stack([thres, youden], axis=-1)
        self.fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                      showlegend=False,
                                      hoverinfo='none',
                                      marker={'color': 'black' if not '_dark' in template_name else 'white'},
                                      opacity=0.2,
                                      mode='lines'))
        best_youden_idx = np.argmax(youden)
        optimal_tpr = decision_dict['TPR'][best_youden_idx]
        optimal_fpr = decision_dict['FPR'][best_youden_idx]
        self.fig.add_trace(go.Scatter(x=[0, optimal_fpr, optimal_fpr], y=[optimal_tpr, optimal_tpr, 0],
                                      hoverinfo='none',
                                      showlegend=False,
                                      mode='lines',
                                      line={'color': 'rgba(245, 66, 66, 0.4)',
                                            'width': 1}))
        self.fig.add_trace(go.Scatter(x=[optimal_fpr], y=[optimal_tpr],
                                      hovertemplate=f"<b>Optimal Threshold: {decision_dict['op']} {thres[best_youden_idx]:.2f}" + "</b><br><br>FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>",
                                      showlegend=False,
                                      marker={'color': 'rgba(245, 66, 66, 0.5)'}))
        if applied_filter_dict is not None:
            op_thres_text = ', '.join([f'{op_th[0]} {op_th[1]:.2f}' for op_th in applied_filter_dict['Operations']])
            tp, fp = applied_filter_dict['TPR'], applied_filter_dict['FPR']
            self.fig.add_trace(go.Scatter(x=[0., fp ,fp], y=[tp, tp, 0.],
                                          hoverinfo='none',
                                          showlegend=False,
                                          mode='lines',
                                          line={'color': 'rgba(9, 155, 9, 0.4)',
                                                'width': 1.5}))
            self.fig.add_trace(go.Scatter(x=[fp], y=[tp],
                                          hovertemplate=f'<b>Applied Threshold: {op_thres_text}</b><br><br>FPR: {fp:.2f}<br>TPR: {tp:.2f}<extra></extra>',
                                          showlegend=False,
                                          marker={'color': 'rgba(9, 155, 9, 0.5)'}))
        self.fig.add_trace(go.Scatter(x=decision_dict['FPR'], y=decision_dict['TPR'],
                                      hovertemplate="<b>Threshold: %{customdata[0]:.2f}</b><br><br>FPR: %{x:.2f}<br>TPR: %{y:.2f}<br>Youden's Index: %{customdata[1]:.2f}<extra></extra>",
                                      customdata=customdata,
                                      fill='tozeroy',
                                      fillcolor='rgba(101, 110, 242, 0.1)',
                                      showlegend=False,
                                      mode='lines',
                                      line={'color': 'rgb(101, 110, 242)', 'width': 2.5}))
        self.fig.add_trace(go.Scatter())
        if auc >= 0.5:
            x_pos, y_pos = 0.86, 0.10
        else:
            x_pos, y_pos = 0.15, 0.95
        self.fig.add_annotation(x=x_pos, y=y_pos,
                                text=f"AUC: {auc:.4f}<br>Best Youden's Index: {max(youden):.4f}<br>Optimal Threshold: {decision_dict['op']} {thres[best_youden_idx]:.2f}",
                                showarrow=False,
                                hovertext="""Metrics<br>"Optimal Threshold" determined by corresponding Youden's Index.""",
                                font={'size': 12},
                                bgcolor='white' if not '_dark' in template_name else 'black',
                                bordercolor='black' if not '_dark' in template_name else 'white',
                                borderpad=4,
                                borderwidth=1,
                                align='left',
                                opacity=0.75)
        if template_name == 'presentation':
            l, r, t, b = 60, 10, 40, 40
        else:
            l, r, t, b = 10, 10, 40, 25
        self.fig['layout'].update({'margin': {'l': l, 'r': r, 't': t, 'b': b},
                                   'title' : f'{column_name} ROC Curve ({decision_dict['op']})',
                                   'template': template_name,
                                   'modebar_add': ['togglespikelines']})
        self.fig['layout']['xaxis']['title']['text'] = 'FPR'
        self.fig['layout']['yaxis']['title']['text'] = 'TPR'
        self.fig['layout']['xaxis']['range'] = [-0.01, 1.01]
        self.fig['layout']['yaxis']['range'] = [-0.01, 1.01]
        config = copy.deepcopy(self.plotly_html_config)
        config['toImageButtonOptions']['filename'] = f'{column_name}_ROC'
        self.fig.write_html(temp_pth, config)
        
    def plot_scatter_threshold_plot(self, applied_filter_dict: dict, column_name: str, temp_pth):
        self.setup_background_color(applied_filter_dict['template'])
        self.fig = go.Figure()
        self.fig.add_vline(x=applied_filter_dict['energy_threshold'],
                           line_width=1.2,
                           line_color=applied_filter_dict['energy_thres_color'], opacity=0.3, 
                           showlegend=True, name='Energy Threshold', legendgroup='threshold',
                           legendgrouptitle_text='Thresholds')
        
        if not applied_filter_dict['list_operations'][0]:
            pass
        elif len(applied_filter_dict['list_operations']) == 1:
            op, thres = applied_filter_dict['list_operations'][0]
            n = 2 if isinstance(thres, float) else 0
            line_width = 7 if op in ['>', '≥'] else 2
            self.fig.add_hline(y=thres,
                               line_width=1.2, line_dash=f'{line_width}, 3', line_color=applied_filter_dict['y_thres_color'],
                               opacity=0.7, showlegend=True, name=f'Threshold ({op} {thres:.{n}f})', legendgroup='threshold')
        else:
            op1, thres_1 = applied_filter_dict['list_operations'][0]
            op2, thres_2 = applied_filter_dict['list_operations'][1]
            n = 2 if isinstance(thres_1, float) else 0
            line_width_1 = 7 if op1 in ['>', '≥'] else 2
            line_width_2 = 7 if op2 in ['>', '≥'] else 2
            self.fig.add_hline(y=thres_1,
                               line_width=1.2, line_dash=f'{line_width_1}, 3', line_color=applied_filter_dict['y_thres_color'],
                               opacity=0.7, showlegend=True, name=f'Threshold ({op1} {thres_1:.{n}f})', legendgroup='threshold')
            self.fig.add_hline(y=thres_2,
                        line_width=1.2, line_dash=f'{line_width_2}, 3', line_color=applied_filter_dict['y_thres_color'],
                        opacity=0.7, showlegend=True, name=f'Threshold ({op2} {thres_2:.{n}f})', legendgroup='threshold')
            
        for k, plot_dict in applied_filter_dict.items():
            if isinstance(plot_dict, dict):
                self.fig.add_trace(go.Scatter(x=plot_dict['x'], y=plot_dict['y'],
                                              mode='markers',
                                              name=f'{k.capitalize()} (N={plot_dict['sum']})',
                                              legendgroup='points',
                                              legendgrouptitle_text='Values',
                                              hovertemplate="<b>Name: %{customdata}</b><br><br>Energy: %{x:.2f}<br>" + column_name + ": %{y:.2f}<extra></extra>",
                                              customdata=applied_filter_dict['Name'],
                                              marker={'color': plot_dict['color'], 'opacity': 0.5}))
            
        self.fig['layout'].update({'margin': {'l': 10, 'r': 10, 't': 40, 'b': 25},
                                   'title' : f'Energy v.s. {column_name}<br><sup>TPR: {applied_filter_dict['tpr_str']}, FPR: {applied_filter_dict['fpr_str']}</sup>',
                                   'template': applied_filter_dict['template'],
                                   'legend': {'orientation': 'v',
                                              'yanchor': 'top',
                                              'xanchor': 'right',
                                              'x': 0.99, 'y': 0.99,
                                              'groupclick': 'toggleitem',
                                              'bgcolor': applied_filter_dict['legend_bgcolor'],
                                              'tracegroupgap': 10,},
                                   'newselection': {'line': {'color': applied_filter_dict['energy_thres_color'], 'dash': '1, 1'}},
                                   'clickmode': 'select'},)
        
        self.fig['layout']['xaxis']['title']['text'] = 'Energy'
        self.fig['layout']['yaxis']['title']['text'] = column_name
        config = copy.deepcopy(self.plotly_html_config)
        config['toImageButtonOptions']['filename'] = f'{column_name}_AutoThreshold_Scatter'
        self.fig.write_html(temp_pth, config)

class AdBlockerInterceptor(QWebEngineUrlRequestInterceptor):
    def __init__(self):
        super().__init__()
    
    def interceptRequest(self, info):
        global rules
        url = info.requestUrl().toString()
        if rules is None:
            with open(rules_pth) as f:
                r = f.readlines()
            rules = AdblockRules(r)
        if rules.should_block(url):
            info.block(True)
    
class SearchDBBrowserWindow(QDialog):
    def __init__(self, parent, url: str, js: str, smiles: str):
        super().__init__(parent)
        
        self.adblocker = AdBlockerInterceptor()
        self.profile = QWebEngineProfile()
        self.profile.setUrlRequestInterceptor(self.adblocker)
        
        self.js = js
        self.smiles = smiles
        self.first_loading = True
        
        self.page = QWebEnginePage(self.profile)
        self.browser = QWebEngineView()
        self.browser.setPage(self.page)
        self.browser.setUrl(QUrl(url))
        self.browser.loadFinished.connect(self.signal_browsers)
        
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_widget.setLayout(toolbar_layout)
        
        # back_action = QAction("Back", self)
        # back_action.triggered.connect(self.browser.back)
        # toolbar.addAction(back_action)
        
        # forward_action = QAction("Forward", self)
        # forward_action.triggered.connect(self.browser.forward)
        # toolbar.addAction(forward_action)
        
        reload_toolbtn = QToolButton(self)
        reload_action = QAction("Reload", self)
        reload_action.triggered.connect(self.browser.reload)
        reload_toolbtn.setDefaultAction(reload_action)
        toolbar_layout.addWidget(reload_toolbtn)
        toolbar_layout.setSpacing(1)
        
        stop_toolbtn = QToolButton(self)
        stop_action = QAction("Stop", self)
        stop_action.triggered.connect(self.browser.stop)
        stop_toolbtn.setDefaultAction(stop_action)
        toolbar_layout.addWidget(stop_toolbtn)
        
        window_layout = QVBoxLayout()
        window_layout.addWidget(toolbar_widget, 1, Qt.AlignmentFlag.AlignLeft)
        window_layout.addWidget(self.browser, 29)
        window_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(window_layout)
        
        self.setWindowTitle("Database Search")
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.75, screen_size.height() * 0.8)
        rect = self.frameGeometry()
        center = QGuiApplication.primaryScreen().availableGeometry().center()
        rect.moveCenter(center)
        self.move(rect.topLeft())
        self.show()
    
    def signal_browsers(self):
        if not self.first_loading:
            return
        self.browser.page().runJavaScript(self.js.format(smiles=self.smiles))
        self.first_loading = False
        
    def closeEvent(self, event):
        self.browser.setPage(None)
        self.browser.deleteLater()
        self.page.deleteLater()
        self.profile.deleteLater()
        self.adblocker.deleteLater()
        
        super().closeEvent(event)

class _ShopperURLBrowser(QDialog):
    def __init__(self, parent, url: str):
        super().__init__(parent)
        
        self.adblocker = AdBlockerInterceptor()
        self.profile = QWebEngineProfile()
        self.profile.setUrlRequestInterceptor(self.adblocker)
        
        self.browser = QWebEngineView()
        self.page = QWebEnginePage(self.profile)
        self.browser.setPage(self.page)
        self.browser.setUrl(QUrl(url))
        self.browser.urlChanged.connect(self.update_url_bar)
        self.browser.loadProgress.connect(self.update_progress)
        self.browser.loadFinished.connect(self.finish_loading)
        
        self.url_bar = QLineEdit(self)
        self.url_bar.setText(url)
        self.url_bar.returnPressed.connect(self.load_url_from_bar)
        
        self.back_button = QPushButton('Back')
        self.forward_button = QPushButton('Forward')
        self.reload_button = QPushButton('Reload')
        self.stop_button = QPushButton('Stop')
        
        self.back_button.clicked.connect(self.browser.back)
        self.forward_button.clicked.connect(self.browser.forward)
        self.reload_button.clicked.connect(self.browser.reload)
        self.stop_button.clicked.connect(self.browser.stop)
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)
        
        nav_bar = QHBoxLayout()
        nav_bar.addWidget(self.back_button)
        nav_bar.addWidget(self.forward_button)
        nav_bar.addWidget(self.reload_button)
        nav_bar.addWidget(self.stop_button)
        nav_bar.addWidget(self.url_bar)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(nav_bar)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.browser)
        self.setLayout(main_layout)
        
        self.setWindowTitle('Supplier Browser')
        self.resize(1150, 750)
        
        rect = self.frameGeometry()
        center = QGuiApplication.primaryScreen().availableGeometry().center()
        rect.moveCenter(center)
        self.move(rect.topLeft())
        
        self.show()
    
    def load_url_from_bar(self):
        url_text = self.url_bar.text()
        if not url_text.startswith('http'):
            url_text = 'http://' + url_text
        self.browser.setUrl(QUrl(url_text))
        
    def update_url_bar(self, qurl):
        self.url_bar.setText(qurl.toString())
        
    def update_progress(self, progress):
        self.progress_bar.setValue(progress)
        
    def finish_loading(self, success):
        if success:
            self.progress_bar.setValue(100)
        else:
            self.url_bar.setText("Failed to load page")
    
    def closeEvent(self, event):
        self.browser.setPage(None)
        self.browser.deleteLater()
        self.page.deleteLater()
        self.profile.deleteLater()
        self.adblocker.deleteLater()
        
        super().closeEvent(event)

class BrowserWithTabs(QDialog):
    closed = Signal()
    
    def __init__(self, parent, url: str):
        super().__init__(parent)
        
        self.tab_browser = TabBrowser(self, url)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.tab_browser)
        self.setLayout(main_layout)
        
        self.setWindowTitle('MolDocker Browser')
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.85, screen_size.height() * 0.9)
        
        rect = self.frameGeometry()
        center = QGuiApplication.primaryScreen().availableGeometry().center()
        rect.moveCenter(center)
        self.move(rect.topLeft())
        self.show()
        
    def closeEvent(self, event):
        for idx in range(self.tab_browser.count()):
            browser_tab: BrowserTab = self.tab_browser.widget(idx)
            browser_tab.clear_browser_resources()
        self.closed.emit()
        super().closeEvent(event)

class TabBrowser(QTabWidget):
    def __init__(self, parent, url):
        super().__init__(parent)
        self.my_colab_notebooks = ['https://colab.research.google.com/github/Ezra-Nemo/Colab_Notebooks/blob/testing/UniDock.ipynb',
                                   'https://colab.research.google.com/github/Ezra-Nemo/Colab_Notebooks/blob/testing/diffdock.ipynb']
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.close_url_tab)
        
        self.profile = QWebEngineProfile("TabbedBrowser", self)
        self.adblocker = AdBlockerInterceptor()
        self.profile.setUrlRequestInterceptor(self.adblocker)
        self.profile.setPersistentStoragePath(cookies_pth)
        self.profile.setPersistentCookiesPolicy(QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies)
        
        # self.setTabBar(ElidedTabBar())
        tab_bar = self.tabBar()
        # tab_bar.setMaximumWidth(250)
        tab_bar.setExpanding(False)
        tab_bar.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        tab_bar.setDocumentMode(True)
        tab_bar.setElideMode(Qt.TextElideMode.ElideRight)
        tab_bar.setMovable(True)
        tab_bar.setUsesScrollButtons(True)
        tab_bar.setStyleSheet("""
                              QTabBar::tab:!selected {
                                  color: #818181;
                              }
                              QTabBar::tab {
                                  max-width: 200px;
                              }""")
        
        self.add_tab_button = QPushButton("+")
        self.add_tab_button.clicked.connect(self.add_new_tab)
        self.setCornerWidget(self.add_tab_button)
        self.add_tab_button.setAutoDefault(False)
        self.add_new_tab(url)
        
    def add_new_tab(self, url: str | bool):
        if not url:
            url = 'https://www.google.com'
        if url in self.my_colab_notebooks:
            browser_tab = AutomaticColabBrowserTab(self, url, self.profile)
        else:
            browser_tab = BrowserTab(self, url, self.profile)
        idx = self.addTab(browser_tab, 'Loading...')
        self.setCurrentIndex(idx)
        
    def close_url_tab(self, idx: int):
        browser_tab: BrowserTab = self.widget(idx)
        browser_tab.clear_browser_resources()
        self.removeTab(idx)
        if self.currentIndex() == -1:
            self.profile.deleteLater()
            self.parent().close()
        
class BrowserTab(QWidget):
    def __init__(self, parent, url, profile):
        super().__init__()
        self.tab: QTabWidget = parent
        self.profile = profile
        self.page = QWebEnginePage(self.profile, self)
        
        self.browser = QWebEngineView()
        self.browser.setPage(self.page)
        self.browser.setUrl(url)
        self.browser.urlChanged.connect(self.update_url_bar)
        self.browser.loadProgress.connect(self.update_progress)
        self.browser.loadFinished.connect(self.finish_loading)
        self.browser.titleChanged.connect(self.update_tab_title)
        self.browser.iconChanged.connect(self.update_tab_icon)
        self.browser.page().createWindow = self.create_new_tab
        settings = self.browser.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, True)
        
        self.url_bar = QLineEdit(self)
        self.url_bar.setText(url)
        self.url_bar.setCursorPosition(0)
        self.url_bar.returnPressed.connect(self.load_url_from_bar)
        
        self.back_button = QPushButton('Back')
        self.forward_button = QPushButton('Forward')
        self.reload_button = QPushButton('Reload')
        self.stop_button = QPushButton('Stop')
        self.back_button.setAutoDefault(False)
        self.forward_button.setAutoDefault(False)
        self.reload_button.setAutoDefault(False)
        self.stop_button.setAutoDefault(False)
        
        self.back_button.clicked.connect(self.browser.back)
        self.forward_button.clicked.connect(self.browser.forward)
        self.reload_button.clicked.connect(self.browser.reload)
        self.stop_button.clicked.connect(self.browser.stop)
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(3)
        
        nav_bar = QHBoxLayout()
        nav_bar.addWidget(self.back_button)
        nav_bar.addWidget(self.forward_button)
        nav_bar.addWidget(self.reload_button)
        nav_bar.addWidget(self.stop_button)
        nav_bar.addWidget(self.url_bar)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(nav_bar)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.browser)
        main_layout.setContentsMargins(5, 0, 5, 0)
        
        self.setLayout(main_layout)
    
    def load_url_from_bar(self):
        url_text = self.url_bar.text()
        if not url_text.startswith('http'):
            url_text = 'http://' + url_text
        self.browser.setUrl(QUrl(url_text))
        
    def update_url_bar(self, q):
        self.url_bar.setText(q.toString())
        self.url_bar.setCursorPosition(0)
        self.url_bar.clearFocus()
        
    def update_progress(self, progress):
        self.progress_bar.setValue(progress)
        
    def finish_loading(self):
        self.progress_bar.setValue(100)
        
    def clear_browser_resources(self):
        self.browser.setPage(None)
        self.page.deleteLater()
    
    def update_tab_title(self, title: str):
        idx = self.tab.indexOf(self)
        self.tab.setTabText(idx, title)
    
    def update_tab_icon(self, icon):
        idx = self.tab.indexOf(self)
        self.tab.setTabIcon(idx, icon)
    
    def create_new_tab(self, windowType):
        if windowType == QWebEnginePage.WebBrowserTab:
            new_tab = BrowserTab(self.tab, "about:blank", self.profile)
            idx = self.tab.addTab(new_tab, "Loading...")
            self.tab.setCurrentIndex(idx)
            return new_tab.browser.page()
        return None

class AutomaticColabBrowserTab(BrowserTab):
    def __init__(self, parent, url, profile):
        self.original_url = url
        google_login = 'https://accounts.google.com/'
        super().__init__(parent, google_login, profile)
        self.browser.urlChanged.connect(self.redirect_after_login)
        self.browser.loadFinished.connect(self.automatically_click_colab)
        self.at_google_site = False
        self.already_redirected = False
        
    def redirect_after_login(self):
        if not self.already_redirected and 'https://myaccount.google.com' in self.browser.url().toString():
            self.already_redirected = True
            self.at_google_site = True
            self.browser.setUrl(self.original_url)
    
    def automatically_click_colab(self):
        if 'UniDock.ipynb' not in self.browser.url().toString():
            self.browser.page().runJavaScript("""
                                              const ctrlDownEvent = new KeyboardEvent('keydown', {
                                                  key: 'Control',
                                                  code: 'ControlLeft',
                                                  keyCode: 17,
                                                  ctrlKey: true,
                                                  bubbles: true,
                                              });
                                              const f9DownEvent = new KeyboardEvent('keydown', {
                                                  key: 'F9',
                                                  code: 'F9',
                                                  keyCode: 120,
                                                  ctrlKey: true,
                                                  bubbles: true,
                                              });
                                              document.dispatchEvent(ctrlDownEvent);
                                              document.dispatchEvent(f9DownEvent);
                                              """)
        else:
            self.browser.page().runJavaScript("""
                                              const ctrlDownEvent = new KeyboardEvent('keydown', {
                                                  key: 'Control',
                                                  code: 'ControlLeft',
                                                  keyCode: 17,
                                                  ctrlKey: true,
                                                  bubbles: true,
                                              });
                                              const f9DownEvent = new KeyboardEvent('keydown', {
                                                  key: 'F8',
                                                  code: 'F8',
                                                  keyCode: 119,
                                                  ctrlKey: true,
                                                  bubbles: true,
                                              });
                                              document.dispatchEvent(ctrlDownEvent);
                                              document.dispatchEvent(f9DownEvent);
                                              """)

class FPocketSignal(QObject):
    sendFPocketData = Signal(str, str)
    jsExecutionDone = Signal()

class FPocketBrowser(QDialog):
    def __init__(self, parent, pdb_file: str | None=None):
        super().__init__(parent)
        self.pdb_file = pdb_file
        self.next_page = False
        self.signal = FPocketSignal()
        self.signal.jsExecutionDone.connect(self.clean_and_close_browser)
        self.is_closing = False
        self.pdb = None
        
        self.browser = QWebEngineView()
        self.browser.setPage(JSLogWebEnginePageSuppress(self))
        curr_dir = os.path.dirname(__file__)
        html_pth = os.path.join(curr_dir, 'fpocketweb', 'index.html')
        self.browser.setUrl(QUrl.fromLocalFile(html_pth))
        self.browser.loadFinished.connect(self.signal_browsers)
        
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_widget.setLayout(toolbar_layout)
        
        window_layout = QVBoxLayout()
        window_layout.addWidget(self.browser)
        window_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(window_layout)
        
        self.setWindowTitle("FPocketWeb")
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.75, screen_size.height() * 0.8)
        rect = self.frameGeometry()
        center = QGuiApplication.primaryScreen().availableGeometry().center()
        rect.moveCenter(center)
        self.move(rect.topLeft())
        self.show()
    
    def signal_browsers(self):
        if self.pdb_file is not None and not self.next_page:
            hm = json.dumps(self.pdb_file)
            self.browser.page().runJavaScript(f"""
                                            var fileInput = document.querySelector('input[type="file"]');
                                            var file = new File([`{hm}`], "protein.pdb", {{ type: "text/plain" }});
                                            var dataTransfer = new DataTransfer();
                                            dataTransfer.items.add(file);
                                            fileInput.files = dataTransfer.files;
                                            var event = new Event('change');
                                            fileInput.dispatchEvent(event);
                                            function checkAndClickButton() {{ 
                                                var btn = document.querySelector('button.btn:nth-child(5)');
                                                if (btn) {{  
                                                    if (!btn.disabled) {{  
                                                        btn.click();
                                                    }} else {{
                                                        setTimeout(function(){{
                                                            checkAndClickButton();
                                                        }}, 1000);
                                                    }}
                                                }}
                                            }}
                                            checkAndClickButton();
                                            """)
        self.next_page = True
        
    def get_pdb_from_web(self):
        js_code = """
            (function() {
                var pdbText = document.querySelector('#textarea');
                return pdbText ? pdbText.value : null;
            })();
        """
        self.browser.page().runJavaScript(js_code, 0, self.retrieve_fpocket_pdb)
        
    def get_table_from_web(self):
        js_code = """
            (function() {
                return document.querySelector('#infoTable').innerText;
            })();
        """
        self.browser.page().runJavaScript(js_code, 0, self.retrieve_fpocket_table)
        
    def retrieve_fpocket_pdb(self, pdb: str):
        if pdb.strip():
            self.pdb = pdb
            self.get_table_from_web()
        else:
            self.signal.jsExecutionDone.emit()
    
    def retrieve_fpocket_table(self, table_text: str):
        self.signal.sendFPocketData.emit(self.pdb, table_text)
        self.signal.jsExecutionDone.emit()
        
    def closeEvent(self, event):
        if not self.is_closing:
            self.is_closing = True
            self.get_pdb_from_web()
            event.ignore()
        else:
            self.signal.jsExecutionDone.disconnect(self.clean_and_close_browser)
            if self.browser:
                self.browser.setPage(None)
                self.browser.deleteLater()
                self.browser = None
            self.parent().fpocket_browser.deleteLater()
            self.parent().fpocket_browser = None
            event.accept()
        
    def clean_and_close_browser(self):
        if not self.is_closing:
            self.is_closing = True
        self.close()

class DocumentationWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Documentation")
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.85, screen_size.height() * 0.85)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.web_view = QWebEngineView()
        
        doc_pth = os.path.join(os.path.dirname(__file__), 'site', 'index.html')
        self.web_view.setUrl(QUrl.fromLocalFile(doc_pth))
        
        layout.addWidget(self.web_view)
        self.setLayout(layout)

class BatchInteractionCalculator(QDialog):
    def __init__(self, mdlname_pdbcombiner_map: dict, parent=None):
        super().__init__(parent)
        curr_dir = os.path.dirname(__file__)
        html_pth = os.path.join(curr_dir, 'ngl_web', 'batch_interaction.html')


class LigPlotWidget(QWidget):
    def __init__(self, complex_pdb: str, interact_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Dialog)
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.75, screen_size.height() * 0.85)
        
        fig = create_ligplot_figure(complex_pdb, interact_df)
        html_pth = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plot_empty', '_tmp_ligplot.html')
        fig.write_html(html_pth, config={'showTips': False, 'displaylogo': False,
                                         'toImageButtonOptions': {'format': 'png', 'scale' : 3},})
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl.fromLocalFile(html_pth))
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.browser)
        
        self.setLayout(layout)
        self.setWindowTitle('LigPlot Beta')