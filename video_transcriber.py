import os
import moviepy.editor as mp
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper

def extract_audio_from_video(video_path, audio_output_path):
    """
    从视频文件中提取音频。
    """
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_output_path)
        print(f"音频已成功提取到: {audio_output_path}")
        return True
    except Exception as e:
        print(f"提取音频时发生错误: {e}")
        return False

def transcribe_audio_with_whisper(audio_path, model_name="base"):
    """
    使用 OpenAI Whisper 模型将音频转换为文本。
    model_name 可以是 "tiny", "base", "small", "medium", "large"。
    更大的模型准确度更高，但需要更多的计算资源和时间。
    """
    try:
        print(f"正在加载 Whisper 模型: {model_name}...")
        model = whisper.load_model(model_name)
        print(f"正在转录音频: {audio_path}...")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"转录音频时发生错误: {e}")
        return None

def split_and_transcribe_audio(audio_path, output_dir="audio_chunks", min_silence_len=500, silence_thresh=-40, model_name="base"):
    """
    将音频文件分割成小块，然后逐块转录。
    这对于长音频文件或内存受限的情况非常有用。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        audio = AudioSegment.from_file(audio_path)
        chunks = split_on_silence(audio,
                                  min_silence_len=min_silence_len,  # 识别为静音的最小长度 (毫秒)
                                  silence_thresh=silence_thresh,    # 低于此分贝值的被认为是静音
                                  keep_silence=200                  # 保留静音的前后200毫秒
                                 )

        full_text = []
        model = whisper.load_model(model_name)
        print(f"正在加载 Whisper 模型: {model_name}...")

        for i, chunk in enumerate(chunks):
            chunk_filename = os.path.join(output_dir, f"chunk_{i}.wav")
            chunk.export(chunk_filename, format="wav")
            print(f"正在转录分块 {i+1}/{len(chunks)}: {chunk_filename}")
            
            # 使用 Whisper 转录每个分块
            result = model.transcribe(chunk_filename)
            text = result["text"]
            full_text.append(text)
            print(f"分块 {i+1} 转录完成。")
            
            # 立即删除已处理的分块文件以节省空间
            try:
                os.remove(chunk_filename)
                print(f"✓ 已删除分块文件: {chunk_filename}")
            except Exception as e:
                print(f"⚠️  删除分块文件时出错: {e}")

        return " ".join(full_text)
    except Exception as e:
        print(f"分段和转录音频时发生错误: {e}")
        return None

def get_filename_without_extension(file_path):
    """
    获取文件名（不包含扩展名和路径）
    """
    base_name = os.path.basename(file_path)  # 获取文件名（包含扩展名）
    filename_without_ext = os.path.splitext(base_name)[0]  # 去除扩展名
    return filename_without_ext

def process_video_to_text(video_path, model_name="base", use_segmentation=False):
    """
    处理视频文件，提取音频并转录为文本
    
    Args:
        video_path: 视频文件路径
        model_name: Whisper模型名称 ("tiny", "base", "small", "medium", "large")
        use_segmentation: 是否使用分段转录（推荐用于长视频）
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return False
    
    # 生成输出文件名
    video_name = get_filename_without_extension(video_path)
    audio_file = f"{video_name}_temp_audio.mp3"  # 临时音频文件
    text_output_file = f"{video_name}.txt"  # 输出文本文件
    
    print(f"正在处理视频: {video_path}")
    print(f"输出文本文件将保存为: {text_output_file}")
    
    # 1. 从视频中提取音频
    if not extract_audio_from_video(video_path, audio_file):
        print("音频提取失败，无法继续处理。")
        return False
    
    # 2. 转录音频
    transcribed_text = None
    
    if use_segmentation:
        print("\n--- 正在使用分段转录 ---")
        transcribed_text = split_and_transcribe_audio(
            audio_file,
            output_dir=f"{video_name}_audio_chunks",
            min_silence_len=700,
            silence_thresh=-35,
            model_name=model_name
        )
    else:
        print("\n--- 正在使用直接转录 ---")
        transcribed_text = transcribe_audio_with_whisper(audio_file, model_name=model_name)
    
    # 3. 保存转录结果
    if transcribed_text:
        print("\n--- 转录完成 ---")
        print("转录结果预览:")
        print(transcribed_text[:200] + "..." if len(transcribed_text) > 200 else transcribed_text)
        
        try:
            with open(text_output_file, "w", encoding="utf-8") as f:
                f.write(transcribed_text)
            print(f"\n转录文本已成功保存到: {text_output_file}")
        except Exception as e:
            print(f"保存文件时发生错误: {e}")
            return False
    else:
        print("转录失败。")
        return False
    
    # 4. 清理临时文件
    try:
        if os.path.exists(audio_file):
            os.remove(audio_file)
            print(f"已删除临时音频文件: {audio_file}")
        
        # 如果使用了分段转录，清理分块目录
        if use_segmentation:
            chunks_dir = f"{video_name}_audio_chunks"
            if os.path.exists(chunks_dir):
                import shutil
                shutil.rmtree(chunks_dir)
                print(f"已删除临时音频分块目录: {chunks_dir}")
    except Exception as e:
        print(f"清理临时文件时发生警告: {e}")
    
    return True

# --- 主要执行部分 ---
if __name__ == "__main__":
    # 获取用户输入的视频路径
    video_path = input("请输入视频文件路径: ").strip().strip('"\'')  # 去除可能的引号
    
    if not video_path:
        print("错误：未提供视频文件路径")
        exit(1)
    
    # 选择Whisper模型
    print("\n可用的Whisper模型:")
    print("1. tiny - 最快，准确度最低")
    print("2. base - 平衡速度和准确度（推荐）")
    print("3. small - 较好准确度")
    print("4. medium - 高准确度")
    print("5. large - 最高准确度，需要更多资源")
    
    model_choice = input("\n请选择模型 (1-5，默认为2): ").strip()
    
    model_map = {
        "1": "tiny",
        "2": "base", 
        "3": "small",
        "4": "medium",
        "5": "large"
    }
    
    selected_model = model_map.get(model_choice, "base")
    print(f"已选择模型: {selected_model}")
    
    # 选择转录方式
    segmentation_choice = input("\n是否使用分段转录？(适合长视频，y/N): ").strip().lower()
    use_segmentation = segmentation_choice in ['y', 'yes', '是']
    
    # 开始处理
    success = process_video_to_text(
        video_path=video_path,
        model_name=selected_model,
        use_segmentation=use_segmentation
    )
    
    if success:
        print("\n🎉 视频转录完成！")
    else:
        print("\n❌ 视频转录失败。")