def getChannelInArray(channel_name, fig):
    rows = 5
    cols = 8
    channel_num = int(channel_name[-3:])
    if channel_num < 7:
        spind = channel_num + 1
    elif channel_num < 33:
        spind = channel_num + 2
    elif channel_num < 65:
        spind = channel_num - 30
    elif channel_num < 97:
        spind = channel_num - 62
    elif channel_num < 119:
        spind = channel_num - 94
    else:
        spind = channel_num - 93

    return fig.add_subplot(rows, cols, spind)
