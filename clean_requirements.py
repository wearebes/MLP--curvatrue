# 读取 conda 导出的文件
with open('conda_env_jq.txt', 'r') as f:
    lines = f.readlines()

# 核心包列表 - 需要转换为版本范围
core_packages = {'torch', 'torchvision', 'numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-learn', 'h5py', 'tqdm', 'pyyaml', 'joblib', 'threadpoolctl'}

# conda 系统包（跳过）
system_packages = {'bzip2', 'ca-certificates', 'openssl', 'ncurses', 'readline', 'sqlite', 'tk', 'xz', 'zlib', 'ld_impl_linux-64', 'expat', 'pthread-stubs', 'libexpat', 'libffi', 'libgcc', 'libgcc-ng', 'libgomp', 'libstdcxx', 'libstdcxx-ng', 'libuuid', 'libxcb', 'libzlib', '_openmp_mutex', 'packaging'}

result = []
for line in lines:
    line = line.strip()
    
    # 跳过空行和注释
    if not line or line.startswith('#'):
        continue
    
    # 跳过包含 @ 的本地路径行
    if '@' in line:
        continue
    
    # 跳过 conda 系统包（包含 =h开头的）
    if '=h' in line or '=main' in line or '=conda' in line:
        continue
    
    # 跳过特定的系统包
    pkg_name = line.split('=')[0] if '=' in line else line
    if pkg_name in system_packages:
        continue
    
    # 跳过 nvidia 相关包
    if pkg_name.startswith('nvidia-'):
        continue
    
    # 处理 pypi 包：移除 =pypi_0 后缀
    if '=pypi_0' in line:
        line = line.replace('=pypi_0', '')
    
    # 跳过不包含有效版本的行（如 setuptools=80.10.2=py312）
    if '=' in line:
        parts = line.split('=')
        if len(parts) > 2:
            # 格式如 setuptools=80.10.2=py312h06a4308_0，只取前两部分
            line = parts[0] + '==' + parts[1]
    
    # 处理版本范围转换
    converted = False
    for pkg in core_packages:
        if '=' in line and line.startswith(pkg + '='):
            version = line.split('=')[1]
            # 清理版本号（移除 +cu130 等后缀）
            version_clean = version.split('+')[0]
            parts = version_clean.split('.')
            if len(parts) >= 2:
                major = parts[0]
                minor = parts[1]
                if minor.isdigit():
                    line = f'{pkg}>={version_clean},<{major}.{int(minor)+1}'
                    converted = True
                    break
            if not converted:
                line = f'{pkg}>={version_clean}'
                converted = True
            break
    
    if line and not line.startswith('@') and '=' in line:
        # 将 = 转换为 ==
        if '>=' not in line and '==' not in line:
            parts = line.split('=')
            if len(parts) == 2:
                line = parts[0] + '==' + parts[1]
        result.append(line)

# 排序并写入
result.sort()
with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(result) + '\n')

print(f'已处理 {len(result)} 个包')
