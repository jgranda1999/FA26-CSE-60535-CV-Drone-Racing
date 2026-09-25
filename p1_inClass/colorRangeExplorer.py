# Graduate Computer Vision (CSE 60535), University of Notre Dame
# Based on the color-selection practical by Adam Czajka and Andrey Kuehlkamp.
# Modified by Mariana Fernandez-Espinosa, Fall 2026
"""Photograph your candy, select examples, and manually explore RGB/HSV/Lab thresholds.

Default: webcam preview, s for snapshot, then select candy and background regions.
--demo is a generated practice image, NOT a webcam. --image opens your own photo.
"""
import argparse
import ast
from datetime import datetime
from itertools import product
from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button

SPACES = {
    'RGB': (None, ['Red', 'Green', 'Blue'], [255, 255, 255]),
    'HSV': (cv2.COLOR_RGB2HSV, ['Hue', 'Saturation', 'Value'], [179, 255, 255]),
    'Lab': (cv2.COLOR_RGB2Lab, ['L (lightness)', 'a (green-red)', 'b (blue-yellow)'], [255, 255, 255]),
}


def convert(rgb, space):
    code = SPACES[space][0]
    return rgb.copy() if code is None else cv2.cvtColor(rgb, code)


# STUDENT CODING AREA: edit here, save, then click Apply my code (RGB only).
TRIM_PERCENT = 5  # Percentile method: compare 0, 5, and 10.

def automatic_thresholds(candy_pixels, trim_percent):
    """Return lower=[R_min, G_min, B_min], upper=[R_max, G_max, B_max]."""
    # Complete ONE branch at a time, then change method to test that branch.
    method = 'percentile'  # 'percentile', 'mean_std', or 'median_mad'
    spread = 2.0          # For mean_std / median_mad: compare 1.5, 2.0, 2.5.

    # PROVIDED: keep this preprocessing unchanged for a fair comparison.
    # It keeps the most colorful quarter of the sample, reducing gray pixels.
    # Assumption: the target is strongly colored; this is not for gray objects.
    pixels = np.asarray(candy_pixels, dtype=float)
    maximum = pixels.max(axis=1)
    colorfulness = (maximum - pixels.min(axis=1)) / np.maximum(maximum, 1)
    representative = pixels[colorfulness >= np.percentile(colorfulness, 75)]

    # representative has N rows (pixels) and 3 columns (R, G, B).
    # Use axis=0 to calculate one result per channel. Replace None with code.
    if method == 'percentile':
        # TODO 1: use np.percentile for the lower and upper bounds.
        # Lower percentile: trim_percent. Upper: 100 - trim_percent.
        # At 5, keep the middle 90% PER CHANNEL, not necessarily across all three.
        lower = None
        upper = None
    elif method == 'mean_std':
        # TODO 2: calculate the mean and standard deviation per channel.
        # Bounds: center minus/plus spread times the standard deviation.
        # Useful NumPy operations: np.mean, np.std.
        lower = None
        upper = None
    elif method == 'median_mad':
        # TODO 3: calculate the median per channel, then the median of the
        # absolute deviations from that median (MAD). Use np.median / np.abs.
        # Width = 1.4826 times MAD; bounds = median minus/plus spread times width.
        # The scaling factor does not guarantee a particular coverage percentage.
        lower = None
        upper = None
    else:
        raise ValueError('Choose percentile, mean_std, or median_mad.')

    # PROVIDED: incomplete branches leave your existing sliders unchanged.
    if lower is None or upper is None:
        raise NotImplementedError(f'Complete the {method} TODO: replace both None values with RGB bounds.')
    return np.clip(lower, 0, 255), np.clip(upper, 0, 255)

# END OF STUDENT CODING AREA. No edits below are needed for this activity.


def run_saved_thresholds(candy_pixels):
    """Reload only the student function and numeric setting, preserving the GUI."""
    path=Path(__file__)
    # Parse source bytes so Python handles UTF-8/BOM independently of Windows locale.
    tree=ast.parse(path.read_bytes(),filename=str(path))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)
               and node.name=='automatic_thresholds']
    if len(functions)!=1:
        raise ValueError('Keep exactly one automatic_thresholds function.')
    settings=[node for node in tree.body if isinstance(node,ast.Assign)
              and any(isinstance(target,ast.Name) and target.id=='TRIM_PERCENT' for target in node.targets)]
    if len(settings)!=1:raise ValueError('Keep one numeric TRIM_PERCENT assignment.')
    trim=ast.literal_eval(settings[0].value)
    if isinstance(trim,bool) or not isinstance(trim,(int,float)) or not 0<=trim<50:
        raise ValueError('TRIM_PERCENT must be a number from 0 to less than 50.')
    namespace=dict(globals(),TRIM_PERCENT=trim)
    # Do not run the rest of the file: that would reopen the camera and explorer.
    module=ast.Module(body=functions,type_ignores=[])
    exec(compile(module,str(path),'exec'),namespace)
    return namespace['automatic_thresholds'](candy_pixels,trim),trim


def effective_ranges(low, high, ignored, space='RGB'):
    low, high = low.copy(), high.copy()
    for i, ignore in enumerate(ignored):
        if ignore:
            low[i], high[i] = 0, SPACES[space][2][i]
    return low, high


def accepted(pixels, low, high, space='RGB'):
    inside = (pixels >= low) & (pixels <= high)
    if space == 'HSV' and low[0] > high[0]:
        inside[..., 0] = (pixels[..., 0] >= low[0]) | (pixels[..., 0] <= high[0])
    return inside.all(axis=-1)


def box_edges(low, high, space='RGB'):
    intervals = [(low, high)]
    if space == 'HSV' and low[0] > high[0]:
        end, start = high.copy(), low.copy()
        end[0], start[0] = 179, 0
        intervals = [(low, end), (start, high)]
    edges = []
    for lo, hi in intervals:
        corners = list(product(*zip(lo.astype(float), hi.astype(float))))
        edges.extend((corners[i], corners[i ^ bit]) for i in range(8)
                     for bit in (1, 2, 4) if i < (i ^ bit))
    return edges


def synthetic_scene(index=0):
    # Clearly labeled drawing for practicing controls, not a real candy photo.
    rng = np.random.default_rng(13+index)
    rgb = np.clip(rng.normal(165, 7, (360, 500, 3)), 0, 255).astype(np.uint8)
    colors = [(225, 60, 55), (65, 205, 75), (65, 85, 225)]
    color = np.array(colors[index % 3])
    yy, xx = np.indices(rgb.shape[:2])
    candy = (xx-150)**2 + (yy-180)**2 < 68**2
    values = np.clip(color + rng.integers(-16, 17, rgb.shape), 0, 255).astype(np.uint8)
    rgb[candy] = values[candy]
    # Colored background patch demonstrates why a channel may matter.
    bg_color = color.copy(); bg_color[index % 3] = 90
    rgb[90:270, 320:465] = np.clip(bg_color + rng.integers(-30, 31, (180,145,3)), 0, 255)
    return rgb, (125, 150, 45, 45), (350, 125, 60, 60)


def open_camera(index=0):
    """Open webcam with V4L2 + MJPEG (needed for WSL/usbipd; YUYV often times out)."""
    camera = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not camera.isOpened():
        return camera
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Warm up: first frames can fail until MJPEG streaming starts.
    for _ in range(5):
        ok, _ = camera.read()
        if ok:
            break
    return camera


def read_rgb(camera):
    bgr = None
    for _ in range(15):
        ok, bgr = camera.read()
        if ok and bgr is not None:
            break
    else:
        raise RuntimeError(
            'Camera unavailable. On WSL, attach the webcam '
            '(usbipd attach --wsl --busid <id>) and ensure no other app is using it.'
        )
    # Keep selection windows manageable without changing aspect ratio.
    scale = min(1.0, 900 / bgr.shape[1], 650 / bgr.shape[0])
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def take_photo(camera):
    print('Show your candy. Focus the webcam window; press s to take a photo, Esc to cancel.')
    try:
        while True:
            rgb = read_rgb(camera)
            preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(preview, (0,0), (preview.shape[1],38), (0,0,0), -1)
            cv2.putText(preview, 'Show candy | S: take photo | Esc: cancel', (10,26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow('1 - Your live webcam', preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('s'), ord('S')):
                return rgb  # Return unannotated pixels, not the preview overlay.
            if key == 27:
                return None
    finally:
        cv2.destroyAllWindows()


def choose_regions(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    while True:
        regions = []
        for title in ('2 - Sample ONE COLOR: small patch inside candy, NOT whole object. Enter accepts',
                      '3 - Drag a BACKGROUND box. Enter accepts; c cancels'):
            print(title)
            region = cv2.selectROI(title, bgr)
            cv2.destroyWindow(title)
            if region[2] == 0 or region[3] == 0:
                return None
            regions.append(region)
        x,y,w,h = regions[0]; a,b,c,d = regions[1]
        if max(x,a) < min(x+w,a+c) and max(y,b) < min(y+h,b+d):
            print('The boxes overlap. Choose a candy-only box and a separate background box.')
            continue
        return regions


# Join neighboring matched pixels for boxes; retain the raw mask for teaching.
MIN_DETECTION_AREA = 200


def detection_boxes(mask, min_area=MIN_DETECTION_AREA):
    """Bound each large connected color region, not necessarily a candy object."""
    # A 3x3 closing bridges tiny gaps; it does not widen the RGB thresholds.
    # This is provided display processing, separate from the student function.
    box_mask=cv2.morphologyEx(mask.astype(np.uint8)*255,cv2.MORPH_CLOSE,
                            np.ones((3,3),dtype=np.uint8))
    contours,_=cv2.findContours(box_mask,cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(contour) for contour in contours
            if cv2.contourArea(contour)>=min_area]


def live_detection(camera, low, high, space, saved_stem):
    """Test frozen thresholds on fresh frames; do not estimate or retune them."""
    low,high=low.copy(),high.copy()
    title='Test live - fixed thresholds'
    show_mask=False
    message='Boxes use 3x3 gap cleanup; M shows the unchanged raw threshold mask.'
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title,1200,550)
    try:
        while True:
            rgb=read_rgb(camera)
            mask=accepted(convert(rgb,space),low,high,space)
            detected=rgb.copy()
            detected[~mask]=(detected[~mask].astype(float)*0.15).astype(np.uint8)
            boxes=detection_boxes(mask)
            webcam=rgb.copy()
            for x,y,w,h in boxes:
                # Draw on display copies only, preserving the raw photo and binary mask.
                for display in (webcam,detected):
                    cv2.rectangle(display,(x,y),(x+w-1,y+h-1),(0,255,0),2)
                    cv2.putText(display,'Color match',(x,max(15,y-6)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            right=np.repeat((mask.astype(np.uint8)*255)[...,None],3,axis=2) if show_mask else detected
            panels=cv2.cvtColor(np.concatenate([webcam,right],axis=1),cv2.COLOR_RGB2BGR)
            # A fixed-width view keeps the instructions readable for any camera resolution.
            panels=cv2.resize(panels,(1200,max(1,round(1200*panels.shape[0]/panels.shape[1]))))
            banner=np.zeros((130,1200,3),dtype=np.uint8)
            diagnostic=message
            if mask.mean()>0.5:
                diagnostic='Most of the scene matches. Return; select a small ONE-COLOR sample or tighten bounds.'
            elif not boxes:
                diagnostic='No large color region: M shows raw matches. Return to widen bounds or sample the new lighting.'
            lines=[f'{space} thresholds FIXED: min {low.tolist()}   max {high.tolist()}',
                   'LEFT: webcam + boxes | RIGHT: '+('black/white mask' if show_mask else 'detected colors')+
                   f' | Regions: {len(boxes)} | Selected: {100*mask.mean():.1f}% of image (not accuracy)',
                   ('M: show detected colors' if show_mask else 'M: show black/white mask')+
                   ' | S: save test evidence | Esc / Q: return to explorer',diagnostic]
            for i,line in enumerate(lines):
                cv2.putText(banner,line,(12,25+30*i),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
            view=np.concatenate([banner,panels],axis=0)
            cv2.imshow(title,view)
            key=cv2.waitKey(20) & 0xFF
            if key in (27,ord('q'),ord('Q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
                break
            if key in (ord('m'),ord('M')):
                show_mask=not show_mask
            if key in (ord('s'),ord('S')):
                stem=Path(str(saved_stem)+'_live_'+datetime.now().strftime('%H%M%S_%f'))
                for suffix,pixels in [('_photo.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)),
                                      ('_mask.png',mask.astype(np.uint8)*255),('_view.png',view)]:
                    if not cv2.imwrite(str(stem)+suffix,pixels):
                        raise OSError('Could not save live evidence: '+str(stem))
                stem.with_suffix('.json').write_text(json.dumps(dict(
                    mode='live_fixed_thresholds',space=space,low=low.tolist(),high=high.tolist(),
                    boxes_xywh=boxes,min_detection_area=MIN_DETECTION_AREA,box_cleanup="3x3 closing",
                    explorer_settings=str(Path(saved_stem).with_suffix('.json'))),indent=2)+'\n')
                message='Saved live photo, mask and view. Now try a different condition; S saves again.'
    finally:
        # Also clean up after a camera disconnect or closing the live window's X.
        try:cv2.destroyWindow(title)
        except cv2.error:pass


def explore(rgb, candy_roi, background_roi, photo_callback=None, synthetic=False,
            select_callback=choose_regions, output_dir=None, initial_space='RGB', live_callback=None):
    state = dict(frame=rgb.copy(), reference=rgb.copy(), rois=[candy_roi,background_roi],
                 space=initial_space, newer=False, show_mask=False, demo_index=0, changed_boundary=None, plot_view="3D",
                 range_source={name:dict(mode='manual',trim_percent=None) for name in SPACES},
                 manual_ranges={name: [np.zeros(3,dtype=int),np.array(spec[2])]
                                for name,spec in SPACES.items()})
    output_dir = Path(output_dir or Path(__file__).parent/'captures')
    fig = plt.figure(figsize=(13.5, 9.5))
    fig.suptitle('Exploring Color Spaces and Object Detection', fontsize=18, y=0.98)
    if synthetic:
        fig.text(0.5,0.94,'PRACTICE DRAWING — no webcam. Run without --demo to photograph your candy.',
                 ha='center',fontsize=10)
    cube = fig.add_axes([0.025,0.355,0.51,0.515],projection='3d')
    hist_axes=[fig.add_axes([0.08,y,0.43,0.12],visible=False) for y in (0.73,0.55,0.37)]
    hist_caption=fig.text(0.08,0.87,'',fontsize=9,visible=False)
    photo_ax = fig.add_axes([0.60,0.64,0.35,0.25])
    result_ax = fig.add_axes([0.60,0.36,0.35,0.25])
    boundary_text = fig.text(0.065,0.337,'',fontsize=8,color='#9b5400')
    ranges_text = fig.text(0.065,0.315,'',fontsize=9,va='top',family='monospace')
    scores_text = fig.text(0.60,0.315,'',fontsize=9,va='top')
    status = fig.text(0.055,0.012,'',fontsize=9,va='bottom')
    fig.text(0.80,0.235,'Predict → change → explain.\nKeep candy; reject background.\nHover for control hints.',fontsize=9,va='top')
    radio_ax = fig.add_axes([0.055,0.105,0.12,0.105]); radio_ax.set_title('Color space',fontsize=10)
    radio = RadioButtons(radio_ax,list(SPACES),active=list(SPACES).index(initial_space))
    fig.text(0.36,0.243,'Your selected thresholds',ha='center',fontsize=10)
    sliders = []
    short_names = {'RGB':['R','G','B'], 'HSV':['H','S','V'], 'Lab':['L','a','b']}
    for channel in range(3):
        for bound in range(2):
            slider = Slider(fig.add_axes([0.29,0.22-(2*channel+bound)*0.017,0.25,0.013]),
                            f'{short_names[initial_space][channel]} {("min","max")[bound]}',
                            0,SPACES[initial_space][2][channel],
                            valinit=state['manual_ranges'][initial_space][bound][channel],valstep=1)
            slider.label.set_fontsize(9); slider.valtext.set_fontsize(9)
            sliders.append(slider)
    check_ax = fig.add_axes([0.60,0.105,0.175,0.105]); check_ax.set_title('Ignore a dimension',fontsize=10)
    checks = CheckButtons(check_ax,SPACES[initial_space][1],[False]*3)
    buttons = {}
    for name,label,rect in [
        ('photo','Change practice image' if synthetic else 'Take another photo',[0.055,0.045,0.20,0.05]),
        ('select','Choose candy again',[0.27,0.045,0.18,0.05]),
        ('save','Save this view',[0.465,0.045,0.16,0.05]),
        ('reset','Reset ranges',[0.64,0.045,0.135,0.05]),
        ('live','Save & test live',[0.795,0.045,0.17,0.05]),
        ('apply','Apply my code',[0.29,0.10,0.25,0.027]),
        ('view3d','3D view',[0.12,0.905,0.12,0.028]),
        ('viewhist','Histograms',[0.25,0.905,0.12,0.028]),
        ('mask','Show black/white mask',[0.795,0.135,0.17,0.045])]:
        buttons[name] = Button(fig.add_axes(rect),label)
    if photo_callback is None and not synthetic:
        buttons['photo'].label.set_text('Photo loaded from file')
        buttons['photo'].set_active(False)

    if live_callback is None:
        buttons['live'].label.set_text('Live: use webcam mode')
        buttons['live'].set_active(False)
    buttons['live'].ax.set_facecolor('#d9eee2')

    def samples(region):
        x,y,w,h = region
        return state['reference'][y:y+h,x:x+w].reshape(-1,3)

    def update(_=None):
        space = state['space']; names, maximum = SPACES[space][1:]
        sample_rgb = [samples(roi) for roi in state['rois']]
        sample_values = [convert(p.reshape(-1,1,3),space).reshape(-1,3) for p in sample_rgb]
        lo,hi = state['manual_ranges'][space]
        lo,hi = effective_ranges(lo,hi,checks.get_status(),space)
        state['low'],state['high'] = lo,hi
        mask = accepted(convert(state['frame'],space),lo,hi,space)
        state['mask'] = mask
        elev,azim = cube.elev,cube.azim
        cube.clear()
        cube.set(xlim=(0,maximum[0]),ylim=(0,maximum[1]),zlim=(0,maximum[2]),
                 xlabel=f'{names[0]} (0–{maximum[0]})',ylabel=f'{names[1]} (0–{maximum[1]})',
                 zlabel=f'{names[2]} (0–{maximum[2]})',title=f'Your example pixels in {space} coordinates')
        cube.set_box_aspect((1,1,1));cube.view_init(elev,azim)
        for values,colors,marker,label in zip(sample_values,sample_rgb,['o','x'],['Candy pixels','Background pixels']):
            indices = np.linspace(0,len(values)-1,min(300,len(values)),dtype=int)
            cube.scatter(*values[indices].T,c=colors[indices]/255.,marker=marker,s=12,alpha=0.65,label=label)
        for start,end in box_edges(lo,hi,space):
            changed=state['changed_boundary']
            highlight=False
            if changed is not None:
                channel,bound=changed
                value=state['manual_ranges'][space][bound][channel]
                highlight=(not checks.get_status()[channel] and start[channel]==value and end[channel]==value)
            cube.plot(*np.array([start,end]).T,color='#db7900' if highlight else 'black',
                      linewidth=3 if highlight else 1.5)
        boundary_text.set_text('Orange edges = last adjusted boundary (unless ignored)'
                               if state['changed_boundary'] is not None else '')
        cube.legend(loc='upper left',fontsize=8)
        photo_ax.clear();photo_ax.imshow(state['frame']);photo_ax.axis('off')
        photo_ax.set_title('New photo — testing your unchanged thresholds' if state['newer']
                           else 'Your photo and selected example regions',fontsize=10)
        if synthetic and not state['newer']:
            photo_ax.set_title('Generated practice image and example regions',fontsize=10)
        if not state['newer']:
            for roi,label in zip(state['rois'],['Candy sample','Background sample']):
                x,y,w,h=roi
                photo_ax.plot([x,x+w,x+w,x,x],[y,y,y+h,y+h,y],color='white',linewidth=1.5)
                photo_ax.text(x,y-5,label,color='white',fontsize=8,bbox=dict(facecolor='black',pad=1))
        result_ax.clear();result_ax.axis('off')
        if state['show_mask']:
            result_ax.imshow(mask,cmap='gray',vmin=0,vmax=1)
            result_ax.set_title('White = selected; black = rejected',fontsize=10)
        else:
            result = state['frame'].copy()
            result[~mask] = (result[~mask].astype(float)*0.15).astype(np.uint8)
            result_ax.imshow(result)
            result_ax.set_title('Detected color stays bright; other pixels are dimmed',fontsize=10)
        intervals=[]
        for j,name in enumerate(names):
            value=f'{lo[j]} to {hi[j]}'
            if space=='HSV' and j==0 and lo[0]>hi[0]:value=f'{lo[0]}–179 OR 0–{hi[0]} (red wraps)'
            intervals.append(f'{name}: {value}')
        note = {'RGB':'Box = accepted RGB ranges. Drag the plot to rotate.',
                'HSV':'Hue wraps at 179/0; two boxes can mean one hue range.',
                'Lab':'Lab axes use OpenCV 8-bit encoding, not physical units.'}[space]
        ranges_text.set_text('\n'.join(intervals)+'\n'+note)
        rates=[100*accepted(p,lo,hi,space).mean() for p in sample_values]
        state['rates']=rates
        scores_text.set_text(f'In your selected example regions:\nCandy pixels kept: {rates[0]:.1f}%\n'
                             f'Background pixels wrongly kept: {rates[1]:.1f}%\n'
                             'These are not accuracy scores for a new photo.')
        refresh_histograms()
        fig.canvas.draw_idle()

    def sync_sliders():
        space = state['space']
        for index,slider in enumerate(sliders):
            channel,bound = divmod(index,2)
            slider.eventson=False
            slider.drawon=False
            slider.valmax=SPACES[space][2][channel]
            slider.ax.set_xlim(0,slider.valmax)
            slider.label.set_text(f'{short_names[space][channel]} {("min","max")[bound]}')
            slider.set_val(int(state['manual_ranges'][space][bound][channel]))
            slider.eventson=True
            slider.drawon=True

    def change_threshold(index,value):
        channel,bound = divmod(index,2)
        space=state['space']; low,high=state['manual_ranges'][space]
        old=int((low,high)[bound][channel]); before=state['rates'][:]
        value=int(value)
        # Hue alone may wrap through zero. All other minimums must be <= maximums.
        if not (space=='HSV' and channel==0):
            value=min(value,int(high[channel])) if bound==0 else max(value,int(low[channel]))
        (low,high)[bound][channel]=value
        state['range_source'][space]=dict(mode='manual',trim_percent=None)
        sync_sliders()
        state['changed_boundary']=(channel,bound)
        update()
        name=SPACES[space][1][channel]
        effect='Ignored channel: this slider currently has no effect.' if checks.get_status()[channel] else (
            'Hue can wrap through zero; inspect both boxes.' if space=='HSV' and channel==0 else
            ('Raised min rejects lower values.' if bound==0 and value>old else
             'Lowered min admits lower values.' if bound==0 and value<old else
             'Lowered max rejects higher values.' if bound==1 and value<old else
             'Raised max admits higher values.' if bound==1 and value>old else 'Boundary unchanged.'))
        impact(f'{name} {("min","max")[bound]}: {old} → {value}. {effect}',before)


    def reset(_):
        space=state['space']
        state['changed_boundary']=None
        state['range_source'][space]=dict(mode='manual',trim_percent=None)
        state['manual_ranges'][space]=[np.zeros(3,dtype=int),np.array(SPACES[space][2])]
        sync_sliders()
        status.set_text('Full ranges restored for this space: all pixels pass. Narrow the ranges to select candy.')
        update()

    def switch(space):
        state['space']=space
        state['changed_boundary']=None
        checks.eventson=False
        for i,label in enumerate(checks.labels):
            label.set_text(SPACES[space][1][i])
            if checks.get_status()[i]:checks.set_active(i)
        checks.eventson=True
        sync_sliders()
        status.set_text('Manual ranges restored for this space. Ignored dimensions reset; adjust the six sliders.')
        update()

    def another_photo(_):
        if synthetic:
            state['demo_index']+=1
            image,a,b=synthetic_scene(state['demo_index'])
            state.update(frame=image,reference=image.copy(),rois=[a,b],newer=False)
            status.set_text('New practice drawing and examples; your manual thresholds are unchanged.')
        elif photo_callback is not None:
            try:image=photo_callback()
            except RuntimeError as error:
                status.set_text(str(error));fig.canvas.draw_idle();return
            if image is None:
                status.set_text('Photo canceled; your previous photo is still displayed.');fig.canvas.draw_idle();return
            state.update(frame=image,newer=True)
            status.set_text('New photo; manual thresholds unchanged. Test them before adjusting the sliders.')
        update()

    def reselect(_):
        result=select_callback(state['frame'])
        if result is not None:
            state.update(reference=state['frame'].copy(),rois=list(result),newer=False)
            status.set_text('New examples displayed; manual thresholds unchanged. Use the sliders to adjust them.')
            update()
        else:status.set_text('Selection canceled; previous examples retained.');fig.canvas.draw_idle()

    def save(_):
        output_dir.mkdir(parents=True,exist_ok=True)
        stem=output_dir/datetime.now().strftime('color_exploration_%Y%m%d_%H%M%S_%f')
        fig.savefig(str(stem)+'.png',dpi=140)
        for suffix,pixels in [('_photo.png',cv2.cvtColor(state['frame'],cv2.COLOR_RGB2BGR)),
                              ('_mask.png',state['mask'].astype(np.uint8)*255),
                              ('_examples.png',cv2.cvtColor(state['reference'],cv2.COLOR_RGB2BGR))]:
            if not cv2.imwrite(str(stem)+suffix,pixels):
                raise OSError('Could not save explorer evidence: '+str(stem))
        metadata=dict(space=state['space'],low=state['low'].tolist(),high=state['high'].tolist(),
                      threshold_mode=state['range_source'][state['space']]['mode'],
                      trim_percent=state['range_source'][state['space']]['trim_percent'],
                      manual_low=state['manual_ranges'][state['space']][0].tolist(),
                      manual_high=state['manual_ranges'][state['space']][1].tolist(),
                      ignored=list(map(bool,checks.get_status())),
                      example_regions=state['rois'],new_photo=state['newer'],synthetic=synthetic)
        stem.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        status.set_text('Saved view, photo, mask and settings in captures/.');fig.canvas.draw_idle()
        return stem

    def impact(message,before):
        after=state['rates']
        status.set_text(message+f'\nSample impact: candy kept {after[0]-before[0]:+.1f} percentage points; '
                        f'background kept {after[1]-before[1]:+.1f} points. Explain the tradeoff.')
        fig.canvas.draw_idle()

    def ignore_dimension(label):
        before=state['rates'][:]
        update()
        channel=SPACES[state['space']][1].index(label)
        message=(f'{label} ignored: its range spans the whole axis. Predict which extra pixels pass.'
                 if checks.get_status()[channel] else f'{label} restored: its manual thresholds apply again.')
        impact(message,before)

    def test_live(_):
        if live_callback is None or state.get('live_running',False):return
        state['live_running']=True
        widgets=[radio,checks,*sliders,*buttons.values()]
        active=[widget.active for widget in widgets]
        for widget in widgets:widget.active=False
        try:
            stem=save(None)
            status.set_text('Testing fixed thresholds in the webcam window. Esc returns here to revise them.')
            fig.canvas.draw_idle()
            live_callback(state['low'].copy(),state['high'].copy(),state['space'],stem)
            status.set_text('Back from live: what failed under movement or lighting? Adjust ONE boundary and test again.')
        except (RuntimeError,OSError,cv2.error) as error:
            status.set_text('Live test stopped: '+str(error)[:150])
        finally:
            state['live_running']=False
            for widget,was_active in zip(widgets,active):widget.active=was_active
        fig.canvas.draw_idle()

    # Hover hints use a stable footer so they do not cover pixels or move the controls.
    hover_state=dict(axis=None,previous='')
    def hover(event):
        if state.get('live_running',False):return
        hints={radio.ax:'Same pixels, new coordinates. Each space remembers its manual bounds.',
               checks.ax:'Predict first: ignoring a dimension removes that test. It cannot reject extra pixels.',
               cube:'Dots = candy examples; crosses = background. Inside the box = passes all active thresholds.',
               photo_ax:'Small sample rectangles provide comparison pixels; they do not set the manual thresholds.',
               result_ax:'Bright (or white) pixels pass. Look for missing candy AND unwanted background.',
               buttons['view3d'].ax:'See how the three channels occur together in each pixel.',
               buttons['viewhist'].ax:'See each channel separately: candy/background percentages and accepted ranges.',
               buttons['apply'].ax:'Save your function edits, then click to reload and apply them. Your photo and samples stay unchanged.',
               buttons['save'].ax:'Save evidence for your explanation: photo, mask, view and threshold settings.',
               buttons['reset'].ax:'Reset this space to full ranges. Everything passes until you narrow the bounds.',
               buttons['photo'].ax:'Take a new snapshot under changed lighting; thresholds stay fixed.',
               buttons['select'].ax:'Choose new examples for the plot and percentages; thresholds stay fixed.',
               buttons['mask'].ax:'White = accepted; black = rejected. This makes unwanted selection easier to see.',
               buttons['live'].ax:('Save your settings, then test them on live frames. Move candy, change light, remove candy.'
                                  if live_callback is not None else
                                  'Live testing needs webcam mode: run python colorRangeExplorer.py without --demo or --image.')}
        for index,slider in enumerate(sliders):
            channel,bound=divmod(index,2);name=SPACES[state['space']][1][channel]
            hints[slider.ax]=(f'{name} {("minimum","maximum")[bound]}: '+
                             ('raise it to reject lower values; lower it to admit them.' if bound==0 else
                              'lower it to reject higher values; raise it to admit them.'))
            if state['space']=='HSV' and channel==0:
                hints[slider.ax]='Hue is circular: minimum > maximum selects across 179/0. Watch the two accepted boxes.'
            if checks.get_status()[channel]:hints[slider.ax]+=' This dimension is currently ignored.'
        target=event.inaxes if event.inaxes in hints else None
        if target==hover_state['axis']:return
        if hover_state['axis'] is not None and status.get_text().startswith('Hint: '):
            status.set_text(hover_state['previous'])
        hover_state['axis']=target
        if target is not None:
            hover_state['previous']=status.get_text()
            status.set_text('Hint: '+hints[target])
        fig.canvas.draw_idle()

    histogram = dict(axes=hist_axes)

    def refresh_histograms():
        buttons['apply'].active=(state['space']=='RGB' and not state.get('live_running',False))
        buttons['apply'].label.set_text('Apply my code' if state['space']=='RGB' else 'Apply my code (RGB only)')
        if state['plot_view']!='Histograms':return
        space=state['space']; names,maximum=SPACES[space][1:]
        values=[convert(samples(roi).reshape(-1,1,3),space).reshape(-1,3) for roi in state['rois']]
        low,high=state['low'],state['high']
        hist_caption.set_text(f'{space}: each sample totals 100%; shading = accepted values')
        for i,ax in enumerate(histogram['axes']):
            ax.clear()
            for pixels,color,label in zip(values,['#1876b5','#d46b20'],['Candy sample','Background sample']):
                counts=np.bincount(pixels[:,i],minlength=maximum[i]+1)
                ax.plot(np.arange(maximum[i]+1),100*counts/len(pixels),color=color,label=label)
            intervals=[(low[i],high[i])]
            if space=='HSV' and i==0 and low[i]>high[i]:intervals=[(low[i],179),(0,high[i])]
            for left,right in intervals:ax.axvspan(left,right,color='green',alpha=0.10)
            ax.axvline(low[i],color='green',linestyle='--',label=f'Min {low[i]}')
            ax.axvline(high[i],color='purple',linestyle='--',label=f'Max {high[i]}')
            ax.set(xlim=(0,maximum[i]),xlabel=f'{names[i]} value',ylabel='% of sample')
            ax.tick_params(labelsize=8)
            ax.xaxis.label.set_size(9);ax.yaxis.label.set_size(9)
            ax.set_title('Ignored: full range applied' if checks.get_status()[i] else '',fontsize=9)
            ax.legend(loc='upper right',fontsize=8,ncol=2)
    def apply_automatic(_):
        if state['space']!='RGB' or state.get('live_running',False):return
        try:
            # This is the only place the student function is called.
            (lower,upper),applied_trim=run_saved_thresholds(samples(state['rois'][0]))
            lower,upper=np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
            if (lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all()
                    or (lower<0).any() or (upper>255).any() or (lower>upper).any()):
                raise ValueError('Return three finite lower/upper bounds, ordered within 0–255.')
        except Exception as error:
            status.set_text(f'{type(error).__name__}: {str(error)[:100]}\nFix automatic_thresholds, save the file, and click Apply my code again.')
            fig.canvas.draw_idle()
            return
        state['range_source']['RGB']=dict(mode='coded_function',trim_percent=applied_trim)
        state['manual_ranges']['RGB']=[np.floor(lower).astype(int),np.ceil(upper).astype(int)]
        state['changed_boundary']=None
        # Apply all three computed ranges; reset ignored channels explicitly.
        checks.eventson=False
        for i,ignored in enumerate(checks.get_status()):
            if ignored:checks.set_active(i)
        checks.eventson=True
        sync_sliders();update()
        if state['rates'][1]>=30:
            status.set_text(f'Background overlap: {state["rates"][1]:.1f}% of background examples pass.\n'
                            'Choose candy again: select a small ONE-COLOR patch, not the whole object. Then Apply my code.')
        else:
            status.set_text('Reloaded your saved function and applied its bounds. '
                            'Ignored dimensions reset. Compare the result, then test live.')
        fig.canvas.draw_idle()

    def set_plot_view(view):
        state['plot_view']=view
        cube.set_visible(view=='3D')
        for ax in hist_axes:ax.set_visible(view=='Histograms')
        hist_caption.set_visible(view=='Histograms')
        for key,selected in [('view3d',view=='3D'),('viewhist',view=='Histograms')]:
            buttons[key].color='#d9eaf6' if selected else '0.85'
            buttons[key].ax.set_facecolor(buttons[key].color)
        refresh_histograms()
        fig.canvas.draw_idle()

    def toggle(_):
        state['show_mask']=not state['show_mask']
        buttons['mask'].label.set_text('Show detected colors' if state['show_mask'] else 'Show black/white mask')
        update()

    radio.on_clicked(switch);checks.on_clicked(ignore_dimension)
    fig.canvas.mpl_connect('motion_notify_event',hover)
    for index,slider in enumerate(sliders):
        slider.on_changed(lambda value,index=index:change_threshold(index,value))
    for key,callback in [('photo',another_photo),('select',reselect),('save',save),('mask',toggle),('reset',reset),('live',test_live),('apply',apply_automatic),('view3d',lambda _:set_plot_view('3D')),('viewhist',lambda _:set_plot_view('Histograms'))]:
        buttons[key].on_clicked(callback)
    status.set_text('Start: all pixels pass. Narrow the six thresholds to enclose candy pixels and exclude background.')
    update()
    set_plot_view('3D')
    fig.explorer=dict(state=state,radio=radio,sliders=sliders,checks=checks,buttons=buttons,
                      update=update,switch=switch,photo=another_photo,reselect=reselect,save=save,
                      toggle=toggle,reset=reset,cube=cube,live=test_live,hover=hover,status=status,histogram=histogram,set_plot_view=set_plot_view,apply_automatic=apply_automatic)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--demo',action='store_true',help='Generated practice drawing; NOT your webcam')
    source.add_argument('--image',type=Path,help='Select examples from your own image file')
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--space',choices=list(SPACES),default='RGB')
    parser.add_argument('--save',type=Path,help='Save initial figure instead of opening explorer')
    args=parser.parse_args();camera=None
    try:
        if args.demo:
            print('PRACTICE MODE: remove --demo to use your webcam.')
            rgb,a,b=synthetic_scene()
        else:
            if args.image:
                bgr=cv2.imread(str(args.image))
                if bgr is None:raise ValueError('Cannot read photo: '+str(args.image))
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            else:
                camera=open_camera(args.camera)
                if not camera.isOpened():raise RuntimeError('Cannot open camera. Check permissions or --camera 1.')
                rgb=take_photo(camera)
                if rgb is None:return
            regions=choose_regions(rgb)
            if regions is None:
                print('Selection canceled. Run again when ready.');return
            a,b=regions
        fig=explore(rgb,a,b,photo_callback=(lambda:take_photo(camera)) if camera is not None else None,
                    synthetic=args.demo,initial_space=args.space,
                    live_callback=(lambda low,high,space,stem:live_detection(camera,low,high,space,stem))
                    if camera is not None else None)
        if args.save:
            args.save.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(args.save,dpi=140);print('Saved:',args.save)
        else:plt.show()
    finally:
        if camera is not None:camera.release()
        if not args.demo:cv2.destroyAllWindows()
        plt.close('all')


if __name__=='__main__':
    main()
