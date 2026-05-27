# 1. 清空所有代理（干掉报错的 socks://）
unset $(env | grep -i proxy | cut -d '=' -f 1)

# 2. 设置 HTTP/HTTPS 代理（支持 huggingface 下载）
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
