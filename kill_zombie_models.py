#!/usr/bin/env python3
"""
僵尸模型进程清理脚本

用于清理不在 model_pool 管理中但仍在运行的 vLLM 模型进程。
这些进程通常是由于竞态条件、异常退出等原因产生的孤儿进程。

使用方法:
    python kill_zombie_models.py           # 交互模式，显示僵尸进程并询问是否清理
    python kill_zombie_models.py --dry-run # 仅显示僵尸进程，不执行清理
    python kill_zombie_models.py --force   # 强制清理所有僵尸进程，不询问
    python kill_zombie_models.py --all     # 清理所有 vLLM 进程（包括正常运行的）
"""

import os
import sys
import argparse
import getpass
from typing import Optional
from dataclasses import dataclass

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import psutil
except ImportError:
    print("错误: 需要安装 psutil 库")
    print("运行: pip install psutil")
    sys.exit(1)


@dataclass
class VLLMProcess:
    """vLLM 进程信息"""
    pid: int
    port: Optional[int]
    model_name: str
    gpu_ids: Optional[str]
    memory_mb: float
    status: str
    cmdline: str
    is_managed: bool = False  # 是否在 model_pool 中管理


def get_current_user() -> str:
    """获取当前用户名"""
    return getpass.getuser()


def parse_vllm_cmdline(cmdline: list[str]) -> tuple[Optional[str], Optional[int]]:
    """
    解析 vLLM 命令行参数，提取模型名和端口
    
    Returns:
        (model_name, port)
    """
    model_name = None
    port = None
    
    cmdline_str = " ".join(cmdline)
    
    # 提取 --model 参数
    for i, arg in enumerate(cmdline):
        if arg == "--model" and i + 1 < len(cmdline):
            model_name = cmdline[i + 1]
        elif arg.startswith("--model="):
            model_name = arg.split("=", 1)[1]
        elif arg == "--port" and i + 1 < len(cmdline):
            try:
                port = int(cmdline[i + 1])
            except ValueError:
                pass
        elif arg.startswith("--port="):
            try:
                port = int(arg.split("=", 1)[1])
            except ValueError:
                pass
    
    # 如果模型名是路径，提取最后的模型名部分
    if model_name and "/" in model_name:
        # 例如 /path/to/Qwen/Qwen3-8B -> Qwen3-8B
        model_name = model_name.rstrip("/").split("/")[-1]
    
    return model_name, port


def get_managed_pids() -> set[int]:
    """
    获取当前 model_pool 中管理的所有 PID
    
    Returns:
        管理中的 PID 集合
    """
    managed_pids = set()
    
    try:
        from src.web.multi_model_utils import model_pool, model_global_lock
        from src.models.vllm_loader import VLLMServer
        
        with model_global_lock:
            for model_name, entry in model_pool.items():
                if isinstance(entry, dict):
                    server = entry.get("server")
                    if isinstance(server, VLLMServer) and server.pid:
                        managed_pids.add(server.pid)
    except ImportError:
        # 如果无法导入，说明服务可能没运行，返回空集合
        pass
    except Exception as e:
        print(f"警告: 无法获取管理中的 PID: {e}")
    
    return managed_pids


def find_vllm_processes(current_user_only: bool = True) -> list[VLLMProcess]:
    """
    查找所有 vLLM 进程
    
    Args:
        current_user_only: 是否只查找当前用户的进程
        
    Returns:
        VLLMProcess 列表
    """
    current_user = get_current_user()
    managed_pids = get_managed_pids()
    vllm_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cmdline', 'memory_info', 'status']):
        try:
            info = proc.info
            pid = info['pid']
            username = info.get('username', '')
            cmdline = info.get('cmdline') or []
            name = info.get('name', '').lower()
            
            # 过滤条件
            if current_user_only and username != current_user:
                continue
            
            if 'python' not in name:
                continue
            
            cmdline_str = " ".join(cmdline)
            if 'vllm.entrypoints.openai.api_server' not in cmdline_str:
                continue
            
            # 解析命令行
            model_name, port = parse_vllm_cmdline(cmdline)
            
            # 获取内存使用
            memory_mb = 0.0
            try:
                mem_info = info.get('memory_info')
                if mem_info:
                    memory_mb = mem_info.rss / (1024 * 1024)
            except Exception:
                pass
            
            # 获取 GPU 信息（从环境变量）
            gpu_ids = None
            try:
                env = proc.environ()
                gpu_ids = env.get('CUDA_VISIBLE_DEVICES')
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
            
            # 获取进程状态
            status = info.get('status', 'unknown')
            
            vllm_proc = VLLMProcess(
                pid=pid,
                port=port,
                model_name=model_name or "未知模型",
                gpu_ids=gpu_ids,
                memory_mb=memory_mb,
                status=status,
                cmdline=cmdline_str[:200] + "..." if len(cmdline_str) > 200 else cmdline_str,
                is_managed=(pid in managed_pids)
            )
            vllm_processes.append(vllm_proc)
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception as e:
            print(f"警告: 处理进程时出错: {e}")
            continue
    
    return vllm_processes


def print_process_table(processes: list[VLLMProcess], title: str = "vLLM 进程列表"):
    """打印进程表格"""
    if not processes:
        print(f"\n{title}: 无")
        return
    
    print(f"\n{title} (共 {len(processes)} 个):")
    print("-" * 100)
    print(f"{'PID':>8} {'端口':>6} {'GPU':>8} {'内存(MB)':>10} {'状态':>10} {'管理':>6} {'模型名':<30}")
    print("-" * 100)
    
    for proc in processes:
        managed_str = "✓" if proc.is_managed else "✗"
        gpu_str = proc.gpu_ids or "-"
        port_str = str(proc.port) if proc.port else "-"
        print(f"{proc.pid:>8} {port_str:>6} {gpu_str:>8} {proc.memory_mb:>10.1f} {proc.status:>10} {managed_str:>6} {proc.model_name:<30}")
    
    print("-" * 100)


def kill_process(pid: int, force: bool = False) -> bool:
    """
    终止进程
    
    Args:
        pid: 进程 ID
        force: 是否强制终止（SIGKILL）
        
    Returns:
        是否成功
    """
    try:
        proc = psutil.Process(pid)
        
        # 先终止子进程
        children = []
        try:
            children = proc.children(recursive=True)
        except psutil.Error:
            pass
        
        for child in children:
            try:
                if force:
                    child.kill()
                else:
                    child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        if children:
            psutil.wait_procs(children, timeout=3)
        
        # 终止主进程
        if force:
            proc.kill()
        else:
            proc.terminate()
        
        proc.wait(timeout=5)
        return True
        
    except psutil.NoSuchProcess:
        return True  # 进程已不存在
    except psutil.TimeoutExpired:
        # 如果 SIGTERM 超时，尝试 SIGKILL
        try:
            proc.kill()
            proc.wait(timeout=3)
            return True
        except Exception:
            return False
    except Exception as e:
        print(f"终止进程 {pid} 时出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="清理僵尸 vLLM 模型进程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python kill_zombie_models.py           # 交互模式
    python kill_zombie_models.py --dry-run # 仅显示，不清理
    python kill_zombie_models.py --force   # 强制清理所有僵尸进程
    python kill_zombie_models.py --all     # 清理所有 vLLM 进程
        """
    )
    parser.add_argument("--dry-run", action="store_true", help="仅显示僵尸进程，不执行清理")
    parser.add_argument("--force", action="store_true", help="强制清理所有僵尸进程，不询问")
    parser.add_argument("--all", action="store_true", help="清理所有 vLLM 进程（包括正常管理的）")
    parser.add_argument("--kill-9", action="store_true", help="使用 SIGKILL 强制终止")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("       vLLM 僵尸模型进程清理工具")
    print("=" * 60)
    
    # 查找所有 vLLM 进程
    print("\n正在扫描 vLLM 进程...")
    all_processes = find_vllm_processes(current_user_only=True)
    
    if not all_processes:
        print("\n未找到任何 vLLM 进程。")
        return
    
    # 分类
    managed_processes = [p for p in all_processes if p.is_managed]
    zombie_processes = [p for p in all_processes if not p.is_managed]
    
    # 显示所有进程
    print_process_table(managed_processes, "正常管理的进程")
    print_process_table(zombie_processes, "僵尸进程（不在 model_pool 中）")
    
    # 确定要清理的进程
    if args.all:
        targets = all_processes
        print(f"\n[--all 模式] 将清理所有 {len(targets)} 个 vLLM 进程")
    else:
        targets = zombie_processes
        if not targets:
            print("\n没有发现僵尸进程，无需清理。")
            return
        print(f"\n发现 {len(targets)} 个僵尸进程")
    
    # dry-run 模式
    if args.dry_run:
        print("\n[--dry-run 模式] 仅显示，不执行清理。")
        print("要执行清理，请移除 --dry-run 参数。")
        return
    
    # 确认清理
    if not args.force:
        print("\n即将终止以下进程:")
        for proc in targets:
            print(f"  PID {proc.pid}: {proc.model_name}")
        
        confirm = input("\n确认清理这些进程? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("已取消。")
            return
    
    # 执行清理
    print("\n开始清理...")
    success_count = 0
    fail_count = 0
    
    for proc in targets:
        print(f"  终止 PID {proc.pid} ({proc.model_name})...", end=" ")
        if kill_process(proc.pid, force=args.kill_9):
            print("✓ 成功")
            success_count += 1
        else:
            print("✗ 失败")
            fail_count += 1
    
    # 总结
    print("\n" + "=" * 60)
    print(f"清理完成: 成功 {success_count} 个, 失败 {fail_count} 个")
    print("=" * 60)
    
    if fail_count > 0:
        print("\n提示: 对于失败的进程，可以尝试使用 --kill-9 参数强制终止")


if __name__ == "__main__":
    main()

