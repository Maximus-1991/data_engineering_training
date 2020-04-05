# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function
import os
import sys

import airflow
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0, parent_dir)

from src.models.train_model import train_model
from src.data.retrieve_data import load_data
from src.data.generate_dataset import generate_train_test_val_dataset


args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2),
}

dag = DAG(
    dag_id='train_model_python_operator',
    default_args=args,
    schedule_interval=None,
)

# [START howto_operator_python]
download_data_task = PythonOperator(
    task_id='download_data',
    python_callable=load_data,
    op_kwargs={'data_url':"http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"},
    dag=dag,
)
# [END howto_operator_python]

# [START howto_operator_python]
generate_dataset_task = PythonOperator(
    task_id='generate_train_val_test_data',
    python_callable=generate_train_test_val_dataset,
    op_kwargs={'dataset': 'cifar-10-batches-py'},
    dag=dag,
)
# [END howto_operator_python]

# [START howto_operator_python]
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    op_kwargs={},
    dag=dag,
)
# [END howto_operator_python]

download_data_task >> generate_dataset_task >> train_model_task
