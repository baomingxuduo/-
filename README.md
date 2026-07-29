# -
毕业论文代码
# -*- coding: utf-8 -*-
"""
胰腺炎预后预测机器学习模型构建系统
描述: 完整的机器学习建模流程，包含特征工程、模型训练、超参数优化、模型解释。
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import time
from datetime import datetime
import json
import pickle
import joblib
import hashlib
import logging
from collections import Counter, defaultdict
from itertools import combinations, product, chain
from functools import wraps, lru_cache
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import copy
import gc
import random
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
import inspect

# ============================================================================
# 0. 系统初始化
# ============================================================================

APP_NAME = "PancreatitisPredictor"
AUTHOR = "Mingming"

def get_system_info():
    """获取系统信息"""
    import platform
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'machine': platform.machine()
    }

SYSTEM_INFO = get_system_info()

# ============================================================================
# 1. 日志系统配置
# ============================================================================

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class LoggerManager:
    """日志管理器"""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化日志系统"""
        os.makedirs('logs', exist_ok=True)
        self.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S'
        
        # 根日志器
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(self.log_format, self.date_format))
        self.root_logger.addHandler(console_handler)
        
        # 添加文件处理器
        log_file = f'logs/{APP_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(self.log_format, self.date_format))
        self.root_logger.addHandler(file_handler)
        
        # 错误日志单独记录
        error_log_file = f'logs/{APP_NAME}_errors.log'
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(self.log_format, self.date_format))
        self.root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取日志器"""
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]

# 创建全局日志器
logger_manager = LoggerManager()
logger = logger_manager.get_logger(__name__)

# ============================================================================
# 2. 配置管理系统
# ============================================================================
class DataConfig:
    """数据配置"""
    data_file: str = 'pancreatitis_simple_directional_features.csv'
    target_col: Optional[str] = None
    group_col: Optional[str] = '分组'
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    cv_folds: int = 5
    n_jobs: int = -1
    verbose: int = 1

class FeatureEngineeringConfig:
    """特征工程配置"""
    create_interactions: bool = True
    create_ratios: bool = True
    create_polynomial: bool = True
    create_aggregations: bool = True
    create_cross_terms: bool = True
    max_interaction_features: int = 30
    polynomial_degree: int = 2
    n_bins: int = 5
    binning_method: str = 'quantile'  # quantile, uniform, kmeans

class PreprocessingConfig:
    """预处理配置"""
    imputation_strategy: str = 'median'
    scaling_method: str = 'standard'  # standard, robust, minmax, maxabs
    handle_outliers: bool = True
    outlier_method: str = 'iqr'  # iqr, zscore, isolation_forest
    outlier_threshold: float = 1.5
    remove_duplicates: bool = True
    handle_categorical: bool = True
    encoding_method: str = 'onehot'  # onehot, label, target

class FeatureSelectionConfig:
    """特征选择配置"""
    enabled: bool = True
    method: str = 'combined'  # f_classif, mutual_info, rfe, combined
    threshold: float = 0.01
    max_features: int = 30
    min_features: int = 5
    rfe_estimator: str = 'random_forest'
    rfe_step: int = 1

@dataclass
class ImbalanceHandlingConfig:
    """不平衡处理配置"""
    enabled: bool = True
    method: str = 'smote'  # smote, adasyn, smote_tomek, borderlinesmote, svmsmote
    sampling_strategy: str = 'auto'
    random_state: int = 42
    k_neighbors: int = 5
    n_jobs: int = -1

class ModelConfig:
    """模型配置"""
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    tuning: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'n_iter': 20,
        'cv': 3,
        'scoring': 'roc_auc',
        'n_jobs': -1
    })

class VisualizationConfig:
    """可视化配置"""
    enabled: bool = True
    format: str = 'png'
    dpi: int = 300
    figsize: Tuple[int, int] = (12, 8)
    font_family: str = 'Times New Roman'
    save_figures: bool = True
    interactive: bool = False
    plotly_enabled: bool = False

class OutputConfig:
    """输出配置"""
    save_models: bool = True
    save_predictions: bool = True
    save_results: bool = True
    save_config: bool = True
    save_plots: bool = True
    output_dir: str = 'output_results'
    model_dir: str = 'models'
    report_dir: str = 'reports'
    plot_dir: str = 'figures'
    temp_dir: str = 'temp'

class AdvancedConfig:
    """高级配置"""
    calculate_confidence_intervals: bool = True
    n_bootstrap: int = 1000
    calculate_statistical_tests: bool = True
    calculate_feature_importance: bool = True
    calculate_shap: bool = True
    calculate_lime: bool = False
    calculate_partial_dependence: bool = True
    enable_auto_ml: bool = False
    enable_deep_learning: bool = False
    enable_ensemble: bool = True
    enable_model_stacking: bool = False
    enable_online_learning: bool = False
    enable_model_monitoring: bool = True
    enable_model_validation: bool = True

class PipelineConfig:
    """部署配置"""
    data: DataConfig = field(default_factory=DataConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    imbalance_handling: ImbalanceHandlingConfig = field(default_factory=ImbalanceHandlingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    
    # 模型字典
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化模型配置"""
        if not self.models:
            self._init_default_models()
        self._create_directories()
    
    def _init_default_models(self):
        """初始化默认模型配置"""
        # XGBoost
        self.models['xgboost'] = ModelConfig(
            enabled=True,
            params={
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'scale_pos_weight': 1,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'verbosity': 0
            },
            tuning={
                'enabled': True,
                'n_iter': 30,
                'cv': 3,
                'scoring': 'roc_auc',
                'param_grid': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'min_child_weight': [1, 3, 5]
                }
            }
        )
        
        # 随机森林
        self.models['random_forest'] = ModelConfig(
            enabled=True,
            params={
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            },
            tuning={
                'enabled': True,
                'n_iter': 20,
                'cv': 3,
                'scoring': 'roc_auc',
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            }
        )
        
        # 梯度提升
        self.models['gradient_boosting'] = ModelConfig(
            enabled=True,
            params={
                'n_estimators': 150,
                'max_depth': 5,
                'learning_rate': 0.1,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'subsample': 0.8,
                'max_features': 'sqrt',
                'random_state': 42
            },
            tuning={
                'enabled': True,
                'n_iter': 20,
                'cv': 3,
                'scoring': 'roc_auc',
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0]
                }
            }
        )
        
        # AdaBoost
        self.models['adaboost'] = ModelConfig(
            enabled=True,
            params={
                'n_estimators': 200,
                'learning_rate': 0.1,
                'algorithm': 'SAMME.R',
                'random_state': 42
            },
            tuning={
                'enabled': False,
                'n_iter': 10,
                'cv': 3,
                'scoring': 'roc_auc',
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.5]
                }
            }
        )
        
        # SVM
        self.models['svm'] = ModelConfig(
            enabled=True,
            params={
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'probability': True,
                'class_weight': 'balanced',
                'random_state': 42
            },
            tuning={
                'enabled': True,
                'n_iter': 20,
                'cv': 3,
                'scoring': 'roc_auc',
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                }
            }
        )
        
        # MLP
        self.models['mlp'] = ModelConfig(
            enabled=True,
            params={
                'hidden_layer_sizes': (100, 50, 25),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'batch_size': 'auto',
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 1000,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 10,
                'random_state': 42
            },
            tuning={
                'enabled': True,
                'n_iter': 15,
                'cv': 3,
                'scoring': 'roc_auc',
                'param_grid': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'learning_rate_init': [0.0001, 0.001, 0.01],
                    'activation': ['relu', 'tanh', 'logistic']
                }
            }
        )
        
        # 逻辑回归
        self.models['logistic_regression'] = ModelConfig(
            enabled=True,
            params={
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42
            },
            tuning={
                'enabled': True,
                'n_iter': 15,
                'cv': 3,
                'scoring': 'roc_auc',
                'param_grid': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['lbfgs', 'liblinear', 'saga']
                }
            }
        )
        
        # LightGBM
        self.models['lightgbm'] = ModelConfig(
            enabled=False,
            params={
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'num_leaves': 31,
                'min_child_samples': 20,
                'random_state': 42,
                'class_weight': 'balanced',
                'verbosity': -1
            },
            tuning={
                'enabled': True,
                'n_iter': 20,
                'cv': 3,
                'scoring': 'roc_auc',
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'num_leaves': [31, 63, 127],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0]
                }
            }
        )
        
        # CatBoost
        self.models['catboost'] = ModelConfig(
            enabled=False,
            params={
                'iterations': 100,
                'depth': 5,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3,
                'border_count': 254,
                'random_seed': 42,
                'verbose': False
            },
            tuning={
                'enabled': True,
                'n_iter': 20,
                'cv': 3,
                'scoring': 'roc_auc',
                'param_grid': {
                    'iterations': [100, 200, 300],
                    'depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'l2_leaf_reg': [1, 3, 5, 7]
                }
            }
        )
    
    def _create_directories(self):
        """创建输出目录"""
        dirs = [
            self.output.output_dir,
            self.output.model_dir,
            self.output.report_dir,
            self.output.plot_dir,
            self.output.temp_dir,
            'logs'
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_enabled_models(self) -> Dict[str, ModelConfig]:
        """获取启用的模型"""
        return {name: config for name, config in self.models.items() if config.enabled}
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if key == 'models':
                result[key] = {k: asdict(v) for k, v in value.items()}
            elif hasattr(value, '__dataclass_fields__'):
                result[key] = asdict(value)
            else:
                result[key] = value
        return result
    
    def save(self, filepath: str):
        """保存配置"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str) -> 'PipelineConfig':
        """加载配置"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if key == 'models':
                for model_name, model_dict in value.items():
                    if model_name in config.models:
                        for param_key, param_value in model_dict.items():
                            if hasattr(config.models[model_name], param_key):
                                setattr(config.models[model_name], param_key, param_value)
            else:
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config

# 创建全局配置
CONFIG = PipelineConfig()
logger.info(f"配置初始化完成: {CONFIG.output.output_dir}")

# ============================================================================
# 3. 数据验证和质量检查
# ============================================================================

class DataQualityReport:
    """数据质量报告"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report = {}
    
    def generate(self) -> Dict:
        """生成完整的数据质量报告"""
        self.report = {
            'basic_info': self._basic_info(),
            'missing_values': self._missing_values(),
            'duplicates': self._duplicates(),
            'outliers': self._outliers(),
            'correlations': self._correlations(),
            'data_types': self._data_types(),
            'distributions': self._distributions(),
            'class_balance': self._class_balance()
        }
        return self.report
    
    def _basic_info(self) -> Dict:
        """基本信息"""
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'column_names': self.df.columns.tolist()
        }
    
    def _missing_values(self) -> Dict:
        """缺失值分析"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'count': missing,
            'percentage': missing_pct
        })
        missing_df = missing_df[missing_df['count'] > 0]
        
        return {
            'total_missing': missing.sum(),
            'total_missing_pct': (missing.sum() / (len(self.df) * len(self.df.columns))) * 100,
            'by_column': missing_df.to_dict()
        }
    
    def _duplicates(self) -> Dict:
        """重复值分析"""
        return {
            'duplicate_rows': self.df.duplicated().sum(),
            'duplicate_rows_pct': (self.df.duplicated().sum() / len(self.df)) * 100
        }
    
    def _outliers(self) -> Dict:
        """异常值分析"""
        outliers = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            n_outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            if n_outliers > 0:
                outliers[col] = {
                    'count': n_outliers,
                    'percentage': (n_outliers / len(self.df)) * 100,
                    'lower_bound': lower,
                    'upper_bound': upper
                }
        
        return outliers
    
    def _correlations(self) -> Dict:
        """相关性分析"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            
            # 找出高相关特征对
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        high_corr.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr
                        })
            
            return {
                'matrix': corr_matrix.to_dict(),
                'high_correlations': high_corr
            }
        return {}
    
    def _data_types(self) -> Dict:
        """数据类型分析"""
        dtypes = self.df.dtypes
        return {
            'by_type': dtypes.value_counts().to_dict(),
            'details': dtypes.to_dict()
        }
    
    def _distributions(self) -> Dict:
        """分布统计"""
        distributions = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                distributions[col] = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75),
                    'skewness': data.skew(),
                    'kurtosis': data.kurtosis()
                }
        
        return distributions
    
    def _class_balance(self) -> Dict:
        """类别平衡分析"""
        # 查找二分类目标变量
        target_col = None
        for col in self.df.columns:
            if len(self.df[col].dropna().unique()) == 2:
                target_col = col
                break
        
        if target_col:
            counts = self.df[target_col].value_counts()
            return {
                'target_column': target_col,
                'counts': counts.to_dict(),
                'percentages': (counts / len(self.df) * 100).to_dict(),
                'imbalance_ratio': max(counts) / min(counts) if len(counts) > 1 else 1
            }
        return {}

# ============================================================================
# 4. 高级特征工程
# ============================================================================

class AdvancedFeatureEngineer:
    """高级特征工程类"""
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.created_features = []
        self.feature_stats = {}
    
    def engineer_features(self, df: pd.DataFrame, base_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """执行特征工程"""
        logger.info("开始高级特征工程...")
        df_engineered = df.copy()
        all_features = base_features.copy()
        
        # 1. 创建交互特征
        if self.config.create_interactions:
            all_features = self._create_interaction_features(df_engineered, all_features)
        
        # 2. 创建比率特征
        if self.config.create_ratios:
            all_features = self._create_ratio_features(df_engineered, all_features)
        
        # 3. 创建多项式特征
        if self.config.create_polynomial:
            all_features = self._create_polynomial_features(df_engineered, all_features)
        
        # 4. 创建聚合特征
        if self.config.create_aggregations:
            all_features = self._create_aggregation_features(df_engineered, all_features)
        
        # 5. 创建交叉项
        if self.config.create_cross_terms:
            all_features = self._create_cross_terms(df_engineered, all_features)
        
        # 6. 创建统计特征
        all_features = self._create_statistical_features(df_engineered, all_features)
        
        # 7. 创建离散化特征
        all_features = self._create_binned_features(df_engineered, all_features)
        
        # 8. 创建秩特征
        all_features = self._create_rank_features(df_engineered, all_features)
        
        logger.info(f"特征工程完成，新增 {len(self.created_features)} 个特征")
        logger.info(f"总特征数: {len(all_features)}")
        
        return df_engineered, all_features
    
    def _create_interaction_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """创建交互特征"""
        logger.info("创建交互特征...")
        numeric_features = self._get_numeric_features(df, features)
        
        if len(numeric_features) < 2:
            return features
        
        # 选择重要的数值特征
        important_features = self._select_important_features(df, numeric_features)
        n_features = min(len(important_features), 5)
        selected_features = important_features[:n_features]
        
        count = 0
        for feat1, feat2 in combinations(selected_features, 2):
            if count >= self.config.max_interaction_features:
                break
            
            interaction_name = f"{feat1}_x_{feat2}"
            df[interaction_name] = df[feat1] * df[feat2]
            features.append(interaction_name)
            self.created_features.append(interaction_name)
            count += 1
            
            # 添加交互
            if count < self.config.max_interaction_features:
                interaction_name2 = f"{feat1}_2x_{feat2}"
                df[interaction_name2] = df[feat1]**2 * df[feat2]
                features.append(interaction_name2)
                self.created_features.append(interaction_name2)
                count += 1
        
        return features
    
    def _create_ratio_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """创建比率特征"""
        logger.info("创建比率特征...")
        numeric_features = self._get_numeric_features(df, features)
        
        if len(numeric_features) < 2:
            return features
        
        count = 0
        for feat1, feat2 in combinations(numeric_features[:5], 2):
            if count >= self.config.max_interaction_features:
                break
            
            ratio_name = f"{feat1}_div_{feat2}"
            df[ratio_name] = df[feat1] / (df[feat2] + 1e-6)
            features.append(ratio_name)
            self.created_features.append(ratio_name)
            count += 1
          
            if count < self.config.max_interaction_features:
                ratio_name2 = f"{feat2}_div_{feat1}"
                df[ratio_name2] = df[feat2] / (df[feat1] + 1e-6)
                features.append(ratio_name2)
                self.created_features.append(ratio_name2)
                count += 1
        
        return features
    
    def _create_polynomial_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """创建多项式特征"""
        logger.info("创建多项式特征...")
        numeric_features = self._get_numeric_features(df, features)
        
        for feat in numeric_features[:3]:  # 限制数量
            if feat in df.columns:
                # 平方
                square_name = f"{feat}_squared"
                df[square_name] = df[feat] ** 2
                features.append(square_name)
                self.created_features.append(square_name)
                
                # 立方
                cube_name = f"{feat}_cubed"
                if df[feat].abs().max() < 100:
                    df[cube_name] = df[feat] ** 3
                    features.append(cube_name)
                    self.created_features.append(cube_name)
                
                # 对数变换
                if (df[feat] > 0).all():
                    log_name = f"{feat}_log"
                    df[log_name] = np.log1p(df[feat])
                    features.append(log_name)
                    self.created_features.append(log_name)
                
                # 指数变换
                exp_name = f"{feat}_exp"
                if df[feat].abs().max() < 5:
                    df[exp_name] = np.exp(df[feat])
                    features.append(exp_name)
                    self.created_features.append(exp_name)
        
        return features
    
    def _create_aggregation_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """创建聚合特征"""
        logger.info("创建聚合特征...")
        numeric_features = self._get_numeric_features(df, features)
        
        for feat in numeric_features:
            if feat not in df.columns:
                continue
            
            # 归一化
            norm_name = f"{feat}_normalized"
            df[norm_name] = (df[feat] - df[feat].mean()) / (df[feat].std() + 1e-6)
            features.append(norm_name)
            self.created_features.append(norm_name)
            
            # 标准化
            std_name = f"{feat}_standardized"
            df[std_name] = (df[feat] - df[feat].min()) / (df[feat].max() - df[feat].min() + 1e-6)
            features.append(std_name)
            self.created_features.append(std_name)
            
            # 居中
            center_name = f"{feat}_centered"
            df[center_name] = df[feat] - df[feat].mean()
            features.append(center_name)
            self.created_features.append(center_name)
            
            # 缩放
            scale_name = f"{feat}_scaled"
            df[scale_name] = df[feat] / (df[feat].std() + 1e-6)
            features.append(scale_name)
            self.created_features.append(scale_name)
        
        return features
    
    def _create_cross_terms(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """创建交叉项"""
        logger.info("创建交叉项...")
        numeric_features = self._get_numeric_features(df, features)
        
        if len(numeric_features) < 3:
            return features
        
        for feat1, feat2, feat3 in combinations(numeric_features[:4], 3):
            cross_name = f"{feat1}_x_{feat2}_x_{feat3}"
            df[cross_name] = df[feat1] * df[feat2] * df[feat3]
            features.append(cross_name)
            self.created_features.append(cross_name)
        
        return features
    
    def _create_statistical_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """创建统计特征"""
        logger.info("创建统计特征...")
        
        # 行统计
        numeric_cols = self._get_numeric_features(df, features)
        if len(numeric_cols) > 1:
            df['row_sum'] = df[numeric_cols].sum(axis=1)
            features.append('row_sum')
            self.created_features.append('row_sum')
            
            df['row_mean'] = df[numeric_cols].mean(axis=1)
            features.append('row_mean')
            self.created_features.append('row_mean')
            
            df['row_std'] = df[numeric_cols].std(axis=1)
            features.append('row_std')
            self.created_features.append('row_std')
            
            df['row_max'] = df[numeric_cols].max(axis=1)
            features.append('row_max')
            self.created_features.append('row_max')
            
            df['row_min'] = df[numeric_cols].min(axis=1)
            features.append('row_min')
            self.created_features.append('row_min')
            
            df['row_range'] = df['row_max'] - df['row_min']
            features.append('row_range')
            self.created_features.append('row_range')
        
        return features
    
    def _create_binned_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """创建分箱特征"""
        logger.info("创建分箱特征...")
        numeric_features = self._get_numeric_features(df, features)
        
        for feat in numeric_features[:3]:
            if feat not in df.columns:
                continue
            
            if self.config.binning_method == 'quantile':
                try:
                    bins = pd.qcut(df[feat], q=self.config.n_bins, labels=False, duplicates='drop')
                    bin_name = f"{feat}_binned"
                    df[bin_name] = bins
                    features.append(bin_name)
                    self.created_features.append(bin_name)
                except:
                    pass
            elif self.config.binning_method == 'uniform':
                bins = pd.cut(df[feat], bins=self.config.n_bins, labels=False)
                bin_name = f"{feat}_binned"
                df[bin_name] = bins
                features.append(bin_name)
                self.created_features.append(bin_name)
        
        return features
    
    def _create_rank_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """创建秩特征"""
        logger.info("创建秩特征...")
        numeric_features = self._get_numeric_features(df, features)
        
        for feat in numeric_features[:2]:
            if feat not in df.columns:
                continue
            
            rank_name = f"{feat}_rank"
            df[rank_name] = df[feat].rank(pct=True)
            features.append(rank_name)
            self.created_features.append(rank_name)
        
        return features
    
    def _get_numeric_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """获取数值型特征"""
        numeric_features = []
        for feat in features:
            if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
                numeric_features.append(feat)
        return numeric_features
    
    def _select_important_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """选择重要特征"""
        # 使用方差和相关性简单选择
        important = []
        for feat in features:
            if feat in df.columns:
                # 计算方差
                variance = df[feat].var()
                # 计算与目标的相关性（如果有）
                importance_score = variance
                
                important.append((feat, importance_score))
        
        # 按重要性排序
        important.sort(key=lambda x: x[1], reverse=True)
        return [feat for feat, _ in important]

# ============================================================================
# 5. 超参数优化
# ============================================================================

class HyperparameterOptimizer:
    """超参数优化类"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results = {}
        self.best_params = {}
    
    def optimize(self, model, model_name: str, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Dict]:
        """执行超参数优化"""
        model_config = self.config.models.get(model_name)
        if not model_config or not model_config.tuning.get('enabled', False):
            return model, {}
        
        logger.info(f"优化 {model_name} 超参数...")
        
        from sklearn.model_selection import RandomizedSearchCV
        
        param_grid = model_config.tuning.get('param_grid', {})
        if not param_grid:
            return model, {}
        
        try:
            random_search = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=model_config.tuning.get('n_iter', 20),
                cv=model_config.tuning.get('cv', 3),
                scoring=model_config.tuning.get('scoring', 'roc_auc'),
                random_state=self.config.data.random_state,
                n_jobs=self.config.data.n_jobs,
                verbose=0,
                return_train_score=True
            )
            
            random_search.fit(X, y)
            
            self.results[model_name] = {
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'cv_results': random_search.cv_results_
            }
            
            logger.info(f"  {model_name} 最佳得分: {random_search.best_score_:.3f}")
            logger.info(f"  {model_name} 最佳参数: {random_search.best_params_}")
            
            return random_search.best_estimator_, random_search.best_params_
            
        except Exception as e:
            logger.error(f"  {model_name} 优化失败: {e}")
            return model, {}

# ============================================================================
# 6. 模型评估和验证
# ============================================================================

class ModelValidator:
    """模型验证类"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.validation_results = {}
    
    def validate(self, model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """执行交叉验证"""
        from sklearn.model_selection import cross_validate
        
        scoring = {
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc',
            'f1': 'f1_macro',
            'precision': 'precision_macro',
            'recall': 'recall_macro'
        }
        
        try:
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=self.config.data.n_jobs
            )
            
            results = {}
            for key, values in cv_results.items():
                results[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values.tolist()
                }
            
            return results
            
        except Exception as e:
            logger.error(f"交叉验证失败: {e}")
            return {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray) -> Dict:
        """计算评估指标"""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                   f1_score, roc_auc_score, confusion_matrix,
                                   matthews_corrcoef, cohen_kappa_score,
                                   balanced_accuracy_score, log_loss,
                                   brier_score_loss)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'kappa': cohen_kappa_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_prob),
            'brier_score': brier_score_loss(y_true, y_prob)
        }
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 计算额外指标
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        return metrics

# ============================================================================
# 7. 模型解释器
# ============================================================================

class ModelInterpreter:
    """模型解释类"""
    
    def __init__(self, model, feature_names: List[str], X_train: np.ndarray, X_test: np.ndarray):
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        self.X_test = X_test
        self.explanations = {}
    
    def explain(self, methods: List[str] = ['shap', 'feature_importance', 'partial_dependence']) -> Dict:
        """执行模型解释"""
        logger.info("开始模型解释...")
        
        for method in methods:
            if method == 'shap':
                self.explanations['shap'] = self._calculate_shap()
            elif method == 'feature_importance':
                self.explanations['feature_importance'] = self._calculate_feature_importance()
            elif method == 'partial_dependence':
                self.explanations['partial_dependence'] = self._calculate_partial_dependence()
            elif method == 'lime':
                self.explanations['lime'] = self._calculate_lime()
        
        return self.explanations
    
    def _calculate_shap(self) -> Dict:
        """计算SHAP值"""
        try:
            import shap
            
            logger.info("计算SHAP值...")
            
            # 检测模型类型
            model_class = self.model.__class__.__name__.lower()
            
            if any(name in model_class for name in ['xgboost', 'randomforest', 'gradientboosting']):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(self.X_test)
            elif any(name in model_class for name in ['svm', 'logisticregression']):
                explainer = shap.LinearExplainer(self.model, self.X_train)
                shap_values = explainer.shap_values(self.X_test)
            else:
                # 使用KernelExplainer（较慢）
                explainer = shap.KernelExplainer(
                    lambda x: self.model.predict_proba(x)[:, 1],
                    self.X_train[:100]
                )
                shap_values = explainer.shap_values(self.X_test[:100])
            
            # 计算特征重要性
            if isinstance(shap_values, list):
                importance = np.abs(shap_values[1]).mean(axis=0)
            else:
                importance = np.abs(shap_values).mean(axis=0)
            
            return {
                'values': shap_values,
                'importance': dict(zip(self.feature_names, importance))
            }
            
        except ImportError:
            logger.warning("SHAP未安装")
            return {}
        except Exception as e:
            logger.error(f"SHAP计算失败: {e}")
            return {}
    
    def _calculate_feature_importance(self) -> Dict:
        """计算特征重要性"""
        importance = {}
        
        # 树模型特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importance['feature_importances'] = dict(
                zip(self.feature_names, self.model.feature_importances_)
            )
        
        # 系数重要性（线性模型）
        if hasattr(self.model, 'coef_'):
            if len(self.model.coef_.shape) == 2:
                coef = self.model.coef_[0]
            else:
                coef = self.model.coef_
            importance['coefficients'] = dict(
                zip(self.feature_names, coef)
            )
        
        # 排列重要性
        try:
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(
                self.model, self.X_test, y_test,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            importance['permutation'] = dict(
                zip(self.feature_names, perm_importance.importances_mean)
            )
        except:
            pass
        
        return importance
    
    def _calculate_partial_dependence(self) -> Dict:
        """计算偏依赖图"""
        try:
            from sklearn.inspection import partial_dependence
            
            results = {}
            for idx, feature in enumerate(self.feature_names[:5]):  # 限制数量
                try:
                    pdp = partial_dependence(
                        self.model, self.X_test,
                        [idx],
                        kind='average'
                    )
                    results[feature] = {
                        'values': pdp['values'][0].tolist(),
                        'average': pdp['average'][0].tolist()
                    }
                except:
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"偏依赖计算失败: {e}")
            return {}
    
    def _calculate_lime(self) -> Dict:
        """计算LIME解释"""
        try:
            import lime
            import lime.lime_tabular
            
            logger.info("计算LIME解释...")
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                class_names=['Class 0', 'Class 1'],
                mode='classification'
            )
            
            # 解释前几个样本
            explanations = []
            for i in range(min(5, len(self.X_test))):
                exp = explainer.explain_instance(
                    self.X_test[i],
                    lambda x: self.model.predict_proba(x),
                    num_features=10
                )
                explanations.append(exp.as_list())
            
            return {'explanations': explanations}
            
        except ImportError:
            logger.warning("LIME未安装")
            return {}
        except Exception as e:
            logger.error(f"LIME计算失败: {e}")
            return {}

# ============================================================================
# 8. 模型监控
# ============================================================================

class ModelMonitor:
    """模型监控类"""
    
    def __init__(self, model, config: PipelineConfig):
        self.model = model
        self.config = config
        self.metrics_history = defaultdict(list)
        self.drift_detection = {}
    
    def log_metrics(self, metrics: Dict, step: int):
        """记录指标"""
        for key, value in metrics.items():
            self.metrics_history[key].append({
                'step': step,
                'value': value,
                'timestamp': datetime.now().isoformat()
            })
    
    def detect_drift(self, X: np.ndarray, reference_X: np.ndarray) -> Dict:
        """检测数据漂移"""
        try:
            from scipy.stats import ks_2samp
            
            drift_results = {}
            n_features = X.shape[1]
            
            for i in range(min(n_features, 10)):  # 限制特征数
                stat, p_value = ks_2samp(X[:, i], reference_X[:, i])
                drift_results[f'feature_{i}'] = {
                    'ks_statistic': stat,
                    'p_value': p_value,
                    'drift_detected': p_value < 0.05
                }
            
            return drift_results
            
        except Exception as e:
            logger.error(f"漂移检测失败: {e}")
            return {}
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        return {
            'metrics_history': dict(self.metrics_history),
            'drift_detection': self.drift_detection,
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict:
        """生成摘要"""
        summary = {}
        for key, values in self.metrics_history.items():
            if values:
                recent_values = [v['value'] for v in values[-10:]]
                summary[key] = {
                    'current': recent_values[-1] if recent_values else None,
                    'mean': np.mean(recent_values) if recent_values else None,
                    'std': np.std(recent_values) if recent_values else None,
                    'trend': 'increasing' if len(recent_values) > 1 and recent_values[-1] > recent_values[0] else 'decreasing'
                }
        return summary

# ============================================================================
# 9. 可视化引擎
# ============================================================================

class VisualizationEngine:
    """可视化引擎"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.figures = []
    
    def setup_style(self):
        """设置绘图样式"""
        import matplotlib.pyplot as plt
        
        plt.rcParams['font.family'] = self.config.font_family
        plt.rcParams['figure.figsize'] = self.config.figsize
        plt.rcParams['savefig.dpi'] = self.config.dpi
        plt.rcParams['figure.dpi'] = 100
        
        # 设置颜色主题
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_roc_curves(self, models_results: Dict, save_path: Optional[str] = None):
        """绘制ROC曲线"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for idx, (model_name, result) in enumerate(models_results.items()):
            if 'fpr' in result and 'tpr' in result:
                color = self.colors[idx % len(self.colors)]
                ax.plot(result['fpr'], result['tpr'], 
                       color=color, lw=2,
                       label=f"{model_name} (AUC = {result['auc']:.3f})")
        
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.close()
        return fig
    
    def plot_confusion_matrices(self, confusions: Dict, save_path: Optional[str] = None):
        """绘制混淆矩阵"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        n_models = len(confusions)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for idx, (model_name, cm) in enumerate(confusions.items()):
            if idx >= n_rows * n_cols:
                break
            
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Pred 0', 'Pred 1'],
                       yticklabels=['True 0', 'True 1'])
            ax.set_title(model_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('Actual', fontsize=10)
        
        # 隐藏空的子图
        for idx in range(n_models, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.close()
        return fig
    
    def plot_feature_importance(self, importance: Dict, title: str = "Feature Importance",
                               save_path: Optional[str] = None):
        """绘制特征重要性"""
        import matplotlib.pyplot as plt
        
        features = list(importance.keys())
        values = list(importance.values())
        
        # 排序
        sorted_idx = np.argsort(values)
        features = [features[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        # 只显示前15个
        if len(features) > 15:
            features = features[-15:]
            values = values[-15:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.barh(range(len(features)), values, color='steelblue')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.close()
        return fig
    
    def plot_learning_curves(self, train_scores: List, val_scores: List, 
                            train_sizes: List, save_path: Optional[str] = None):
        """绘制学习曲线"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='blue', label='Training')
        ax.fill_between(train_sizes, 
                       np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                       np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                       alpha=0.15, color='blue')
        
        ax.plot(train_sizes, np.mean(val_scores, axis=1), 's-', color='red', label='Validation')
        ax.fill_between(train_sizes,
                       np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                       np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                       alpha=0.15, color='red')
        
        ax.set_xlabel('Training Examples', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.close()
        return fig
    
    def plot_model_comparison(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """绘制模型对比图"""
        import matplotlib.pyplot as plt
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        models = results_df['Model'].tolist()
        n_models = len(models)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(n_models)
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = results_df[f'Test_{metric.capitalize()}'].tolist()
            ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        plt.close()
        return fig

# ============================================================================
# 10. 主流程
# ============================================================================

class PredictionPipeline:
    """预测流程类别"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data = None
        self.features = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.scaler = None
        self.monitor = None
        
    def run(self):
        """运行完整流程"""
        start_time = time.time()
        logger.info("="*80)
        logger.info(f"{APP_NAME} v{VERSION}")
        logger.info("="*80)
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"系统信息: {SYSTEM_INFO}")
        logger.info("="*80)
        
        try:
            # 1. 加载数据
            self._load_data()
            
            # 2. 数据探索
            self._explore_data()
            
            # 3. 数据预处理
            self._preprocess_data()
            
            # 4. 特征工程
            self._feature_engineering()
            
            # 5. 数据分割
            X_train, X_test, X_val, y_train, y_test, y_val = self._split_data()
            
            # 6. 特征选择
            if self.config.feature_selection.enabled:
                X_train, X_test, X_val = self._feature_selection(X_train, X_test, X_val, y_train)
            
            # 7. 数据标准化
            X_train_scaled, X_test_scaled, X_val_scaled, scaler = self._scale_data(
                X_train, X_test, X_val
            )
            self.scaler = scaler
            
            # 8. 处理不平衡
            if self.config.imbalance_handling.enabled:
                X_train_final, y_train_final = self._handle_imbalance(
                    X_train_scaled, y_train
                )
            else:
                X_train_final, y_train_final = X_train_scaled, y_train.values
            
            # 9. 训练模型
            self._train_models(X_train_final, y_train_final, X_test_scaled, y_test, X_val_scaled, y_val)
            
            # 10. 模型集成
            if self.config.advanced.enable_ensemble:
                self._create_ensemble()
            
            # 11. 模型评估
            self._evaluate_models(X_test_scaled, y_test, X_val_scaled, y_val)
            
            # 12. 模型解释
            if self.config.advanced.calculate_shap:
                self._interpret_models(X_train_scaled, X_test_scaled)
            
            # 13. 模型监控
            if self.config.advanced.enable_model_monitoring:
                self._setup_monitoring()
            
            # 14. 保存结果
            self._save_results(X_test_scaled, y_test)
            
            # 15. 生成报告
            self._generate_report(start_time)
            
            # 16. 可视化
            if self.config.visualization.enabled:
                self._visualize_results()
            
            elapsed_time = time.time() - start_time
            logger.info("\n" + "="*80)
            logger.info("流程执行完成!")
            logger.info("="*80)
            logger.info(f"总用时: {elapsed_time:.2f} 秒")
            logger.info(f"使用特征数: {len(self.features)}")
            logger.info(f"训练模型数: {len(self.models)}")
            if self.best_model_name:
                best_auc = self.results[self.best_model_name]['test_metrics']['auc']
                logger.info(f"最佳模型: {self.best_model_name} (AUC = {best_auc:.3f})")
            logger.info("="*80)
            
            return self.results, self.best_model
            
        except Exception as e:
            logger.error(f"流程执行失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _load_data(self):
        """加载数据"""
        logger.info("加载数据...")
        data_file = self.config.data.data_file
        
        encodings = ['utf-8', 'gbk', 'utf-8-sig', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(data_file, encoding=encoding)
                logger.info(f"成功加载数据: {data_file} (编码: {encoding})")
                self.data = df
                return
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"加载失败: {e}")
                continue
        
        raise ValueError(f"无法加载文件: {data_file}")
    
    def _explore_data(self):
        """探索数据"""
        logger.info("探索数据...")
        
        # 生成质量报告
        quality_report = DataQualityReport(self.data)
        report = quality_report.generate()
        
        logger.info(f"数据形状: {report['basic_info']['rows']} x {report['basic_info']['columns']}")
        logger.info(f"内存使用: {report['basic_info']['memory_usage']:.2f} MB")
        
        if report['missing_values']['total_missing'] > 0:
            logger.warning(f"缺失值总数: {report['missing_values']['total_missing']}")
        
        # 确定目标变量
        if self.config.data.target_col is None:
            self.config.data.target_col = self._find_target_column()
        
        logger.info(f"目标变量: {self.config.data.target_col}")
        
        # 确定基础特征
        self.base_features = self._get_base_features()
        logger.info(f"基础特征数: {len(self.base_features)}")
    
    def _find_target_column(self) -> str:
        """查找目标列"""
        # 关键字查找
        keywords = ['death', '死亡', 'outcome', 'status', 'mortality', 'label']
        for col in self.data.columns:
            if any(kw in col.lower() for kw in keywords):
                return col
        
        # 二分类查找
        for col in self.data.columns:
            unique_vals = self.data[col].dropna().unique()
            if len(unique_vals) == 2:
                return col
        
        raise ValueError("无法找到目标变量列")
    
    def _get_base_features(self) -> List[str]:
        """获取基础特征"""
        # 从配置中获取或自动选择
        if hasattr(self.config, 'selected_features') and self.config.selected_features:
            features = [f for f in self.config.selected_features if f in self.data.columns]
        else:
            # 排除目标列和分组列
            exclude_cols = [self.config.data.target_col]
            if self.config.data.group_col:
                exclude_cols.append(self.config.data.group_col)
            
            features = [col for col in self.data.columns if col not in exclude_cols]
        
        return features
    
    def _preprocess_data(self):
        """数据预处理"""
        logger.info("数据预处理...")
        
        df = self.data.copy()
        
        # 删除重复行
        if self.config.preprocessing.remove_duplicates:
            n_duplicates = df.duplicated().sum()
            if n_duplicates > 0:
                df = df.drop_duplicates()
                logger.info(f"删除 {n_duplicates} 个重复行")
        
        # 处理缺失值
        if self.config.preprocessing.imputation_strategy:
            for col in self.base_features:
                if col in df.columns and df[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if self.config.preprocessing.imputation_strategy == 'median':
                            df[col].fillna(df[col].median(), inplace=True)
                        elif self.config.preprocessing.imputation_strategy == 'mean':
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif self.config.preprocessing.imputation_strategy == 'mode':
                            df[col].fillna(df[col].mode()[0], inplace=True)
                        else:
                            df[col].fillna(0, inplace=True)
        
        # 处理异常值
        if self.config.preprocessing.handle_outliers:
            for col in self.base_features:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if self.config.preprocessing.outlier_method == 'iqr':
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - self.config.preprocessing.outlier_threshold * IQR
                        upper = Q3 + self.config.preprocessing.outlier_threshold * IQR
                        
                        outliers = (df[col] < lower) | (df[col] > upper)
                        if outliers.sum() > 0:
                            median = df[col].median()
                            df.loc[outliers, col] = median
                            logger.info(f"处理 {outliers.sum()} 个异常值在列 {col}")
        
        self.data = df
        logger.info("数据预处理完成")
    
    def _feature_engineering(self):
        """特征工程"""
        if not any([self.config.feature_engineering.create_interactions,
                   self.config.feature_engineering.create_ratios,
                   self.config.feature_engineering.create_polynomial]):
            self.features = self.base_features.copy()
            return
        
        engineer = AdvancedFeatureEngineer(self.config.feature_engineering)
        self.data, self.features = engineer.engineer_features(
            self.data, self.base_features
        )
    
    def _split_data(self):
        """数据分割"""
        logger.info("数据分割...")
        
        X = self.data[self.features]
        y = self.data[self.config.data.target_col]
        
        group_col = self.config.data.group_col
        
        if group_col and group_col in self.data.columns:
            # 使用预定义分组
            train_data = self.data[self.data[group_col] == 1]
            test_data = self.data[self.data[group_col] == 2]
            val_data = self.data[self.data[group_col] == 3]
            
            X_train = train_data[self.features]
            y_train = train_data[self.config.data.target_col]
            X_test = test_data[self.features]
            y_test = test_data[self.config.data.target_col]
            X_val = val_data[self.features]
            y_val = val_data[self.config.data.target_col]
            
            logger.info(f"训练集: {len(X_train)} 样本")
            logger.info(f"测试集: {len(X_test)} 样本")
            logger.info(f"验证集: {len(X_val)} 样本")
        else:
            # 随机划分
            from sklearn.model_selection import train_test_split
            
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.4, random_state=self.config.data.random_state,
                stratify=y
            )
            X_test, X_val, y_test, y_val = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=self.config.data.random_state,
                stratify=y_temp
            )
            
            logger.info(f"训练集: {len(X_train)} 样本")
            logger.info(f"测试集: {len(X_test)} 样本")
            logger.info(f"验证集: {len(X_val)} 样本")
        
        # 保存特征名称
        self.feature_names = self.features
        
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def _feature_selection(self, X_train, X_test, X_val, y_train):
        """特征选择"""
        logger.info("执行特征选择...")
        
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier
        
        # 1. 统计检验
        try:
            selector = SelectKBest(f_classif, k='all')
            selector.fit(X_train, y_train)
            
            p_values = selector.pvalues_
            selected = [self.features[i] for i in range(len(p_values)) if p_values[i] < 0.05]
            
            if len(selected) >= self.config.feature_selection.min_features:
                X_train = X_train[selected]
                X_test = X_test[selected]
                X_val = X_val[selected]
                self.features = selected
                logger.info(f"统计检验选择: {len(selected)} 个特征")
        except:
            pass
        
        # 2. RFE
        if len(self.features) > 10:
            try:
                estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                n_features = min(15, len(self.features))
                rfe = RFE(estimator, n_features_to_select=n_features, step=2)
                rfe.fit(X_train, y_train)
                
                selected = [self.features[i] for i in range(len(self.features)) if rfe.support_[i]]
                
                if len(selected) >= self.config.feature_selection.min_features:
                    X_train = X_train[selected]
                    X_test = X_test[selected]
                    X_val = X_val[selected]
                    self.features = selected
                    logger.info(f"RFE选择: {len(selected)} 个特征")
            except:
                pass
        
        # 3. 互信息
        if len(self.features) > 10:
            try:
                selector = SelectKBest(mutual_info_classif, k=min(15, len(self.features)))
                selector.fit(X_train, y_train)
                
                selected = [self.features[i] for i in range(len(self.features)) if selector.get_support()[i]]
                
                if len(selected) >= self.config.feature_selection.min_features:
                    X_train = X_train[selected]
                    X_test = X_test[selected]
                    X_val = X_val[selected]
                    self.features = selected
                    logger.info(f"互信息选择: {len(selected)} 个特征")
            except:
                pass
        
        logger.info(f"最终特征数: {len(self.features)}")
        return X_train, X_test, X_val
    
    def _scale_data(self, X_train, X_test, X_val):
        """数据标准化"""
        logger.info("数据标准化...")
        
        from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
        
        method = self.config.preprocessing.scaling_method
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val)
        
        return X_train_scaled, X_test_scaled, X_val_scaled, scaler
    
    def _handle_imbalance(self, X, y):
        """处理不平衡"""
        logger.info("处理类别不平衡...")
        
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.combine import SMOTETomek
        
        method = self.config.imbalance_handling.method
        
        if method == 'smote':
            sampler = SMOTE(
                random_state=self.config.imbalance_handling.random_state,
                sampling_strategy=self.config.imbalance_handling.sampling_strategy,
                k_neighbors=self.config.imbalance_handling.k_neighbors,
                n_jobs=self.config.imbalance_handling.n_jobs
            )
        elif method == 'adasyn':
            sampler = ADASYN(
                random_state=self.config.imbalance_handling.random_state,
                sampling_strategy=self.config.imbalance_handling.sampling_strategy,
                n_neighbors=self.config.imbalance_handling.k_neighbors,
                n_jobs=self.config.imbalance_handling.n_jobs
            )
        elif method == 'smote_tomek':
            sampler = SMOTETomek(
                random_state=self.config.imbalance_handling.random_state,
                sampling_strategy=self.config.imbalance_handling.sampling_strategy
            )
        else:
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        logger.info(f"原始训练集: {len(X)} 样本")
        logger.info(f"平衡后训练集: {len(X_resampled)} 样本")
        logger.info(f"正例率: {y_resampled.mean():.3f}")
        
        return X_resampled, y_resampled
    
    def _train_models(self, X_train, y_train, X_test, y_test, X_val, y_val):
        """训练所有模型"""
        logger.info("训练模型...")
        
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        
        model_mapping = {
            'xgboost': XGBClassifier,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'adaboost': AdaBoostClassifier,
            'svm': SVC,
            'mlp': MLPClassifier,
            'logistic_regression': LogisticRegression
        }
        
        optimizer = HyperparameterOptimizer(self.config)
        validator = ModelValidator(self.config)
        
        enabled_models = self.config.get_enabled_models()
        
        for model_name, model_config in enabled_models.items():
            if model_name not in model_mapping:
                continue
            
            logger.info(f"\n{'='*50}")
            logger.info(f"训练模型: {model_name}")
            logger.info('='*50)
            
            # 创建模型
            model_class = model_mapping[model_name]
            model = model_class(**model_config.params)
            
            # 超参数优化
            model, best_params = optimizer.optimize(model, model_name, X_train, y_train)
            
            # 训练模型
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            logger.info(f"训练时间: {train_time:.2f} 秒")
            
            # 保存模型
            self.models[model_name] = model
            
            # 评估模型
            y_train_pred = model.predict(X_train)
            y_train_prob = model.predict_proba(X_train)[:, 1]
            
            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)[:, 1]
            
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1]
            
            # 计算指标
            train_metrics = validator.calculate_metrics(y_train, y_train_pred, y_train_prob)
            test_metrics = validator.calculate_metrics(y_test, y_test_pred, y_test_prob)
            val_metrics = validator.calculate_metrics(y_val, y_val_pred, y_val_prob)
            
            # 交叉验证
            if self.config.advanced.enable_model_validation:
                cv_results = validator.validate(model, X_train, y_train)
            else:
                cv_results = {}
            
            # 保存结果
            self.results[model_name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'val_metrics': val_metrics,
                'cv_results': cv_results,
                'train_time': train_time,
                'best_params': best_params,
                'predictions': {
                    'train': y_train_pred,
                    'test': y_test_pred,
                    'val': y_val_pred
                },
                'probabilities': {
                    'train': y_train_prob,
                    'test': y_test_prob,
                    'val': y_val_prob
                }
            }
            
            logger.info(f"训练集: AUC={train_metrics['auc']:.3f}, Acc={train_metrics['accuracy']:.3f}")
            logger.info(f"测试集: AUC={test_metrics['auc']:.3f}, Acc={test_metrics['accuracy']:.3f}")
            logger.info(f"验证集: AUC={val_metrics['auc']:.3f}, Acc={val_metrics['accuracy']:.3f}")
        
        # 选择最佳模型
        self._select_best_model()
    
    def _select_best_model(self):
        """选择最佳模型"""
        if not self.results:
            return
        
        best_auc = -1
        best_name = None
        
        for name, result in self.results.items():
            auc = result['test_metrics']['auc']
            if auc > best_auc:
                best_auc = auc
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        logger.info(f"最佳模型: {best_name} (AUC = {best_auc:.3f})")
    
    def _create_ensemble(self):
        """创建集成模型"""
        try:
            # 选择前3个模型
            sorted_models = sorted(
                self.results.items(),
                key=lambda x: x[1]['test_metrics']['auc'],
                reverse=True
            )[:3]
            
            if len(sorted_models) < 2:
                return
            
            model_list = [self.models[name] for name, _ in sorted_models]
            weights = [result['test_metrics']['auc'] for _, result in sorted_models]
            weights = [w / sum(weights) for w in weights]
            
            from sklearn.ensemble import VotingClassifier
            
            ensemble = VotingClassifier(
                estimators=[(name, self.models[name]) for name, _ in sorted_models],
                voting='soft',
                weights=weights
            )
            
            # 训练集成模型（使用原始训练数据）
            ensemble.fit(X_train_scaled, y_train)
            
            self.models['ensemble'] = ensemble
            self.model_names.append('ensemble')
            
            logger.info(f"创建集成模型 (包含 {len(model_list)} 个模型)")
            
        except Exception as e:
            logger.error(f"集成模型创建失败: {e}")
    
    def _evaluate_models(self, X_test, y_test, X_val, y_val):
        """评估所有模型"""
        logger.info("评估模型...")
        
        for model_name, model in self.models.items():
            if model_name not in self.results:
                continue
            
            result = self.results[model_name]
            
            # 计算置信区间
            if self.config.advanced.calculate_confidence_intervals:
                from sklearn.metrics import roc_auc_score
                
                y_prob = result['probabilities']['test']
                n_bootstrap = self.config.advanced.n_bootstrap
                
                auc_scores = []
                for _ in range(n_bootstrap):
                    indices = np.random.choice(len(y_test), len(y_test), replace=True)
                    if len(np.unique(y_test.iloc[indices])) < 2:
                        continue
                    auc_scores.append(roc_auc_score(y_test.iloc[indices], y_prob[indices]))
                
                if auc_scores:
                    result['auc_ci'] = {
                        'mean': np.mean(auc_scores),
                        'lower': np.percentile(auc_scores, 2.5),
                        'upper': np.percentile(auc_scores, 97.5)
                    }
    
    def _interpret_models(self, X_train, X_test):
        """解释模型"""
        if self.best_model is None:
            return
        
        logger.info("解释最佳模型...")
        
        interpreter = ModelInterpreter(
            self.best_model,
            self.features,
            X_train,
            X_test
        )
        
        explanations = interpreter.explain(['shap', 'feature_importance'])
        
        # 保存解释结果
        if explanations:
            self.results[self.best_model_name]['explanations'] = explanations
        
        # 可视化SHAP
        if 'shap' in explanations:
            import matplotlib.pyplot as plt
            shap_values = explanations['shap']['values']
            
            if shap_values is not None:
                try:
                    import shap
                    
                    plt.figure(figsize=(12, 8))
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[1], X_test, 
                                        feature_names=self.features,
                                        show=False)
                    else:
                        shap.summary_plot(shap_values, X_test,
                                        feature_names=self.features,
                                        show=False)
                    
                    plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    shap_path = f"{self.config.output.plot_dir}/shap_summary.png"
                    plt.savefig(shap_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"SHAP图已保存: {shap_path}")
                    
                except Exception as e:
                    logger.error(f"SHAP可视化失败: {e}")
    
    def _setup_monitoring(self):
        """设置模型监控"""
        if self.best_model is None:
            return
        
        self.monitor = ModelMonitor(self.best_model, self.config)
        logger.info("模型监控已设置")
    
    def _save_results(self, X_test, y_test):
        """保存结果"""
        logger.info("保存结果...")
        
        output_dir = self.config.output.output_dir
        
        # 保存模型
        if self.config.output.save_models:
            model_dir = self.config.output.model_dir
            for model_name, model in self.models.items():
                model_path = f"{model_dir}/{model_name}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"保存模型: {model_path}")
            
            # 保存最佳模型
            if self.best_model:
                best_path = f"{model_dir}/best_model.pkl"
                joblib.dump(self.best_model, best_path)
                logger.info(f"保存最佳模型: {best_path}")
        
        # 保存预测结果
        if self.config.output.save_predictions:
            predictions = {}
            for model_name, result in self.results.items():
                predictions[model_name] = {
                    'train': result['predictions']['train'].tolist(),
                    'test': result['predictions']['test'].tolist(),
                    'val': result['predictions']['val'].tolist()
                }
            
            pred_file = f"{output_dir}/predictions.json"
            with open(pred_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"保存预测结果: {pred_file}")
        
        # 保存结果表格
        if self.config.output.save_results:
            results_rows = []
            for model_name, result in self.results.items():
                row = {
                    'Model': model_name,
                    'Train_Accuracy': result['train_metrics']['accuracy'],
                    'Train_AUC': result['train_metrics']['auc'],
                    'Train_F1': result['train_metrics']['f1'],
                    'Test_Accuracy': result['test_metrics']['accuracy'],
                    'Test_Precision': result['test_metrics']['precision'],
                    'Test_Recall': result['test_metrics']['recall'],
                    'Test_F1': result['test_metrics']['f1'],
                    'Test_AUC': result['test_metrics']['auc'],
                    'Val_Accuracy': result['val_metrics']['accuracy'],
                    'Val_AUC': result['val_metrics']['auc'],
                    'Train_Time': result.get('train_time', 0)
                }
                
                if 'auc_ci' in result:
                    row['AUC_CI_Lower'] = result['auc_ci']['lower']
                    row['AUC_CI_Upper'] = result['auc_ci']['upper']
                
                results_rows.append(row)
            
            results_df = pd.DataFrame(results_rows)
            results_df = results_df.sort_values('Test_AUC', ascending=False)
            
            results_file = f"{output_dir}/model_results.csv"
            results_df.to_csv(results_file, index=False)
            logger.info(f"保存结果: {results_file}")
        
        # 保存配置
        if self.config.output.save_config:
            config_file = f"{output_dir}/config.json"
            self.config.save(config_file)
            logger.info(f"保存配置: {config_file}")
    
    def _generate_report(self, start_time):
        """生成报告"""
        logger.info("生成报告...")
        
        report_dir = self.config.output.report_dir
        report_file = f"{report_dir}/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"{APP_NAME} v{VERSION} - 预测报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总用时: {time.time() - start_time:.2f} 秒\n\n")
            
            f.write("1. 数据信息\n")
            f.write("-"*40 + "\n")
            f.write(f"  数据文件: {self.config.data.data_file}\n")
            f.write(f"  样本数: {len(self.data)}\n")
            f.write(f"  特征数: {len(self.features)}\n")
            f.write(f"  目标变量: {self.config.data.target_col}\n\n")
            
            f.write("2. 模型性能总结\n")
            f.write("-"*40 + "\n")
            
            if self.best_model_name:
                best_result = self.results[self.best_model_name]
                f.write(f"  最佳模型: {self.best_model_name}\n")
                f.write(f"  测试集AUC: {best_result['test_metrics']['auc']:.3f}\n")
                f.write(f"  测试集准确率: {best_result['test_metrics']['accuracy']:.3f}\n")
                f.write(f"  测试集F1: {best_result['test_metrics']['f1']:.3f}\n")
                
                if 'auc_ci' in best_result:
                    f.write(f"  AUC 95% CI: [{best_result['auc_ci']['lower']:.3f}, {best_result['auc_ci']['upper']:.3f}]\n")
            
            f.write("\n3. 所有模型性能\n")
            f.write("-"*40 + "\n")
            
            for model_name, result in sorted(
                self.results.items(),
                key=lambda x: x[1]['test_metrics']['auc'],
                reverse=True
            ):
                f.write(f"\n  {model_name}:\n")
                f.write(f"    训练集AUC: {result['train_metrics']['auc']:.3f}\n")
                f.write(f"    测试集AUC: {result['test_metrics']['auc']:.3f}\n")
                f.write(f"    验证集AUC: {result['val_metrics']['auc']:.3f}\n")
                f.write(f"    测试集准确率: {result['test_metrics']['accuracy']:.3f}\n")
                f.write(f"    测试集F1: {result['test_metrics']['f1']:.3f}\n")
            
            f.write("\n4. 使用的特征\n")
            f.write("-"*40 + "\n")
            for i, feature in enumerate(self.features, 1):
                f.write(f"  {i}. {feature}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("报告结束\n")
            f.write("="*80 + "\n")
        
        logger.info(f"报告已生成: {report_file}")
    
    def _visualize_results(self):
        """可视化结果"""
        logger.info("生成可视化...")
        
        viz_engine = VisualizationEngine(self.config.visualization)
        viz_engine.setup_style()
        
        plot_dir = self.config.output.plot_dir
        
        # 1. ROC曲线
        roc_data = {}
        for model_name, result in self.results.items():
            if 'probabilities' in result:
                roc_data[model_name] = {
                    'fpr': self._calculate_roc(result['probabilities']['test'], y_test),
                    'tpr': self._calculate_roc(result['probabilities']['test'], y_test, return_fpr=False),
                    'auc': result['test_metrics']['auc']
                }
        
        if roc_data:
            fig = viz_engine.plot_roc_curves(roc_data, f"{plot_dir}/roc_curves.png")
        
        # 2. 混淆矩阵
        confusions = {}
        for model_name, result in self.results.items():
            if 'test_metrics' in result and 'confusion_matrix' in result['test_metrics']:
                confusions[model_name] = result['test_metrics']['confusion_matrix']
        
        if confusions:
            fig = viz_engine.plot_confusion_matrices(confusions, f"{plot_dir}/confusion_matrices.png")
        
        # 3. 特征重要性
        if self.best_model and hasattr(self.best_model, 'feature_importances_'):
            importance = dict(zip(self.features, self.best_model.feature_importances_))
            fig = viz_engine.plot_feature_importance(
                importance,
                f"{self.best_model_name} Feature Importance",
                f"{plot_dir}/feature_importance.png"
            )
        
        # 4. 模型对比
        if len(self.results) > 1:
            results_df = pd.DataFrame([{
                'Model': name,
                'Test_Accuracy': result['test_metrics']['accuracy'],
                'Test_Precision': result['test_metrics']['precision'],
                'Test_Recall': result['test_metrics']['recall'],
                'Test_F1': result['test_metrics']['f1'],
                'Test_AUC': result['test_metrics']['auc']
            } for name, result in self.results.items()])
            
            fig = viz_engine.plot_model_comparison(results_df, f"{plot_dir}/model_comparison.png")
        
        logger.info(f"可视化完成，保存至: {plot_dir}")
    
    def _calculate_roc(self, y_prob, y_true, return_fpr=True):
        """计算ROC曲线"""
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        if return_fpr:
            return fpr
        return tpr

# ============================================================================
# 11. 主模型入口
# ============================================================================

def main():
    try:
        # 创建模型
        pipeline = PredictionPipeline(CONFIG)
        
        # 运行模型
        results, best_model = pipeline.run()
        
        # 打印最终结果
        print("\n" + "="*80)
        print("最终结果摘要")
        print("="*80)
        print(f"最佳模型: {pipeline.best_model_name}")
        if pipeline.best_model_name:
            best_result = results[pipeline.best_model_name]
            print(f"测试集AUC: {best_result['test_metrics']['auc']:.3f}")
            print(f"测试集准确率: {best_result['test_metrics']['accuracy']:.3f}")
            print(f"测试集F1: {best_result['test_metrics']['f1']:.3f}")
        print("="*80)
        
        return results, best_model
        
    except Exception as e:
        logger.error(f"模型运行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
