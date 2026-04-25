"""
过滤低质量提取记录的工具
"""

def filter_low_quality_records(records, min_quality_score=2):
    """
    过滤低质量记录
    
    质量评分标准（累加）：
    - 有enzyme_name: +1分
    - 有enzyme_type: +1分
    - 有任一动力学参数(Km/Vmax/kcat): +2分
    - 有降解效率: +1分
    - 有产物信息: +1分
    - 有其他实验条件(pH/温度/时间): +1分
    
    Args:
        records: 记录列表
        min_quality_score: 最低质量分数（建议2-3）
        
    Returns:
        filtered_records: 过滤后的记录
        filtered_out: 被过滤掉的记录
    """
    filtered_records = []
    filtered_out = []
    
    for record in records:
        score = 0
        reasons = []
        
        # 1. 有enzyme_name
        if record.get('enzyme_name'):
            score += 1
        else:
            reasons.append("missing enzyme_name")
        
        # 2. 有enzyme_type
        if record.get('enzyme_type'):
            score += 1
        else:
            reasons.append("missing enzyme_type")
        
        # 3. 有动力学参数
        kinetic_params = ['Km_value', 'Vmax_value', 'kcat_value', 'kcat_Km_value']
        has_kinetics = any(record.get(p) is not None for p in kinetic_params)
        if has_kinetics:
            score += 2
            reasons.append("has kinetic parameters")
        
        # 4. 有降解效率
        if record.get('degradation_efficiency') is not None:
            score += 1
            reasons.append("has degradation efficiency")
        
        # 5. 有产物信息
        if record.get('products') and len(record.get('products', [])) > 0:
            score += 1
            reasons.append("has products")
        
        # 6. 有实验条件
        conditions = ['ph', 'temperature_value', 'reaction_time_value']
        has_conditions = any(record.get(c) is not None for c in conditions)
        if has_conditions:
            score += 1
            reasons.append("has experimental conditions")
        
        # 判断是否保留
        if score >= min_quality_score:
            filtered_records.append(record)
        else:
            filtered_out.append({
                'record': record,
                'score': score,
                'reasons': reasons
            })
    
    return filtered_records, filtered_out

def print_filter_statistics(original_count, filtered_count, filtered_out):
    """
    打印过滤统计信息
    """
    print(f"\n{'='*80}")
    print("🔍 Record Filtering Statistics")
    print(f"{'='*80}")
    print(f"Original records:  {original_count}")
    print(f"Kept records:      {filtered_count} ({filtered_count/original_count*100:.1f}%)")
    print(f"Filtered out:      {len(filtered_out)} ({len(filtered_out)/original_count*100:.1f}%)")
    
    if filtered_out:
        print(f"\n❌ Filtered out examples:")
        for item in filtered_out[:5]:
            record = item['record']
            enzyme = record.get('enzyme_name', 'UNKNOWN')
            substrate = record.get('substrate', 'UNKNOWN')
            source = record.get('source_in_document', {}).get('source_type', 'unknown')
            print(f"   - {enzyme:30s} + {substrate:10s} (score: {item['score']}, source: {source})")
            print(f"     Issues: {', '.join([r for r in item['reasons'] if 'missing' in r])}")


# 示例用法
if __name__ == "__main__":
    # 测试示例
    test_records = [
        # 好记录：有酶名、类型、动力学参数
        {
            'enzyme_name': 'CotA',
            'enzyme_type': 'laccase',
            'substrate': 'AFB1',
            'Km_value': 0.5,
            'Km_unit': 'μM',
            'products': [{'name': 'AFQ1'}]
        },
        # 中等记录：有酶名、类型、降解效率
        {
            'enzyme_name': 'MnP',
            'enzyme_type': 'peroxidase',
            'substrate': 'ZEN',
            'degradation_efficiency': 80.0,
            'products': [{'name': 'C18H22O3'}]
        },
        # 差记录：只有酶名和底物
        {
            'enzyme_name': 'Laccase',
            'substrate': 'AFB1'
        },
        # 非常差的记录：连酶名都没有
        {
            'substrate': 'MDZ'
        }
    ]
    
    filtered, filtered_out = filter_low_quality_records(test_records, min_quality_score=2)
    
    print(f"✅ Kept: {len(filtered)} records")
    print(f"❌ Filtered out: {len(filtered_out)} records")
    
    for item in filtered_out:
        record = item['record']
        print(f"\nFiltered: {record.get('enzyme_name', 'NO_NAME')} + {record.get('substrate')}")
        print(f"  Score: {item['score']}")
        print(f"  Reasons: {item['reasons']}")
