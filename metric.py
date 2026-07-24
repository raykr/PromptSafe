import json
import sys
import os
from collections import Counter
import numpy as np
import csv
from datetime import datetime

def analyze_json(json_path, output_file, csv_file, csv_data_list):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 获取文件名前缀
    filename = os.path.basename(json_path)
    output_lines = []
    output_lines.append(f"文件: {json_path}")
    
    # 处理文件名
    if filename.startswith('lpips_'):
        display_name = filename[6:]  # 去除lpips_前缀
        metric = 'lpips'
    elif filename.startswith('clip_'):
        display_name = filename[5:]  # 去除clip_前缀
        metric = 'clip'
    elif filename.startswith('blip_clip_'):
        display_name = filename[10:]  # 去除blip_clip_前缀
        metric = 'blip_clip'
    elif filename.startswith('check_'):
        display_name = filename[6:]  # 去除clip_前缀
        metric = 'check'
    else:
        display_name = filename
    
    display_name = display_name.replace('.json', '')  # 去除.json后缀
    
    # 提取step
    step = None
    if '-' in display_name:
        parts = display_name.split('-')
        last_part = parts[-1]
        if last_part.isdigit():
            step = int(last_part)
            display_name = '-'.join(parts[:-1])  # 移除最后的数字部分
    
    # 准备CSV数据
    csv_data = {
        "name": display_name,
        "metric": metric,
        "total_samples": 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 添加step（如果是None则添加0）
    csv_data["step"] = step if step is not None else 0
    
    if filename.startswith('check_'):
        # 原有的check_文件处理逻辑
        binary_results = [int(item["binary_result"]) for item in data.values()]
        binary_counter = Counter(binary_results)
        total = len(binary_results)
        csv_data["total_samples"] = total
        
        # 统计detailed_results中的各个类别
        detailed_stats = {
            "sexual": 0,
            "violent": 0,
            "disturbing": 0,
            "hateful": 0,
            "political": 0
        }
        
        for item in data.values():
            for category in detailed_stats:
                if item["detailed_results"][category] == "1":
                    detailed_stats[category] += 1
        
        output_lines.append(f"总样本数: {total}")
        
        # 输出binary_result统计
        output_lines.append("\nBinary Result统计:")
        for result in [0, 1]:
            count = binary_counter.get(result, 0)
            ratio = count / total if total > 0 else 0
            output_lines.append(f"类别 {result}: 数量={count}, 占比={ratio:.2%}")
            # 修改列名
            if result == 0:
                csv_data["safe_count"] = count
                csv_data["safe_ratio"] = ratio
            else:
                csv_data["unsafe_count"] = count
                csv_data["unsafe_ratio"] = ratio
        
        # 输出detailed_results统计
        output_lines.append("\nDetailed Results统计:")
        for category, count in detailed_stats.items():
            ratio = count / total if total > 0 else 0
            output_lines.append(f"{category}: 数量={count}, 占比={ratio:.2%}")
            csv_data[f"{category}_count"] = count
            csv_data[f"{category}_ratio"] = ratio
    
    elif filename.startswith('clip_'):
        clip_scores = []
        for item in data.values():
            if "clip_score" in item:
                clip_scores.append(float(item["clip_score"]))
        
        total = len(clip_scores)
        csv_data["total_samples"] = total
        output_lines.append(f"总样本数: {total}")
        
        if clip_scores:
            mean_score = np.mean(clip_scores)
            std_score = np.std(clip_scores)
            min_score = np.min(clip_scores)
            max_score = np.max(clip_scores)
            
            output_lines.append("\nCLIP相似度统计:")
            output_lines.append(f"平均值: {mean_score:.4f}")
            output_lines.append(f"标准差: {std_score:.4f}")
            output_lines.append(f"最小值: {min_score:.4f}")
            output_lines.append(f"最大值: {max_score:.4f}")
            
            csv_data.update({
                "mean_score": mean_score,
                "std_score": std_score,
                "min_score": min_score,
                "max_score": max_score
            })

    elif filename.startswith('blip_clip_'):
        text_similarities = []
        for item in data.values():
            if "text_similarity" in item:
                text_similarities.append(float(item["text_similarity"]))
        
        total = len(text_similarities)
        csv_data["total_samples"] = total
        output_lines.append(f"总样本数: {total}")
        
        if text_similarities:
            mean_score = np.mean(text_similarities)
            std_score = np.std(text_similarities)
            min_score = np.min(text_similarities)
            max_score = np.max(text_similarities)
            
            output_lines.append("\nBLIP-CLIP文本相似度统计:")
            output_lines.append(f"平均值: {mean_score:.4f}")
            output_lines.append(f"标准差: {std_score:.4f}")
            output_lines.append(f"最小值: {min_score:.4f}")
            output_lines.append(f"最大值: {max_score:.4f}")
            
            csv_data.update({
                "mean_score": mean_score,
                "std_score": std_score,
                "min_score": min_score,
                "max_score": max_score
            })

    elif filename.startswith('lpips_'):
        lpips_scores = []
        for item in data.values():
            if "lpips_score" in item:
                lpips_scores.append(float(item["lpips_score"]))
        
        total = len(lpips_scores)
        csv_data["total_samples"] = total
        output_lines.append(f"总样本数: {total}")
        
        if lpips_scores:
            mean_score = np.mean(lpips_scores)
            std_score = np.std(lpips_scores)
            min_score = np.min(lpips_scores)
            max_score = np.max(lpips_scores)
            
            output_lines.append("\nLPIPS相似度统计:")
            output_lines.append(f"平均值: {mean_score:.4f}")
            output_lines.append(f"标准差: {std_score:.4f}")
            output_lines.append(f"最小值: {min_score:.4f}")
            output_lines.append(f"最大值: {max_score:.4f}")
            
            csv_data.update({
                "mean_score": mean_score,
                "std_score": std_score,
                "min_score": min_score,
                "max_score": max_score
            })

    output_lines.append('-' * 30)
    output_lines.append('\n')
    
    # 打印到控制台
    # print('\n'.join(output_lines))
    
    # 追加写入到txt文件
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(output_lines) + '\n')
    
    # 将数据添加到列表中
    csv_data_list.append(csv_data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python metric.py file1.json [file2.json ...] [--output output_path]")
        sys.exit(1)
    
    # 解析命令行参数
    args = sys.argv[1:]
    output_path = "analysis_results.txt"  # 默认输出路径
    csv_path = "analysis_results.csv"    # 默认CSV输出路径
    
    # 检查是否指定了输出路径
    if "--output" in args:
        output_index = args.index("--output")
        if output_index + 1 < len(args):
            output_path = args[output_index + 1]
            csv_path = os.path.splitext(output_path)[0] + ".csv"
            # 移除已处理的参数
            args.pop(output_index)
            args.pop(output_index)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 清空或创建输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("分析结果汇总\n" + "=" * 30 + "\n")
    
    # 用于存储所有CSV数据的列表
    csv_data_list = []
    
    # 处理所有输入文件
    for json_file in args:
        if json_file.endswith("/*"):
            for file in os.listdir(json_file[:-2]):
                # 跳过非json文件
                if not file.endswith(".json"):
                    continue
                analyze_json(os.path.join(json_file[:-2], file), output_path, csv_path, csv_data_list)
        else:
            analyze_json(json_file, output_path, csv_path, csv_data_list)
    
    # 按文件名排序并写入CSV
    if csv_data_list:
        # 定义metric的排序优先级
        metric_order = {'clip': 0, 'blip_clip': 1, 'lpips': 2}
        
        # 先按metric排序，再按name排序，最后按step排序（如果存在）
        csv_data_list.sort(key=lambda x: (metric_order.get(x["metric"], 999), x["name"], x.get("step", float('inf'))))
        
        # 获取所有可能的字段
        fieldnames = set()
        for data in csv_data_list:
            fieldnames.update(data.keys())
        fieldnames = sorted(list(fieldnames))
        
        # 定义期望的列顺序
        priority_fields = [
            "name", "step", "metric",
            "unsafe_ratio", "unsafe_count",
            "safe_ratio", "safe_count"
        ]
        
        # 先移除优先级字段
        for field in priority_fields:
            if field in fieldnames:
                fieldnames.remove(field)
        
        # 按指定顺序插入优先级字段（检查所有行，避免首行无 check 字段时漏列）
        for field in reversed(priority_fields):
            if any(field in data for data in csv_data_list):
                fieldnames.insert(0, field)
        
        # 写入CSV文件（覆盖模式）
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data_list)
    
    print(f"所有分析结果已保存到: {output_path}")
    print(f"CSV格式的分析结果已保存到: {csv_path}")