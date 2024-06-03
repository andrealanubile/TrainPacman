import * as constants from './constants'
import Animator from './animator'
import Spritesheet from './spritesheet';
import { Sprite } from 'pixi.js'

class PacmanSprites extends Spritesheet {
    constructor() {
        super();
        this.alive = true;
        this.direction = constants.LEFT;
        // this.image = this.getStartImage();
        this.sprite = new Sprite();
        this.sprite.anchor.set(0.5);
        this.animations = new Object()
        this.animations.left = new Animator([[8,0], [0, 0], [0, 2], [0, 0]])
        this.animations.right = new Animator([[10,0], [2, 0], [2, 2], [2, 0]])
        this.animations.up = new Animator([[10,2], [6, 0], [6, 2], [6, 0]])
        this.animations.down = new Animator([[8,2], [4, 0], [4, 2], [4, 0]])
        this.animations.death = new Animator([[0, 12], [2, 12], [4, 12], [6, 12], [8, 12], [10, 12], [12, 12], [14, 12], [16, 12], [18, 12], [20, 12]], 6, false)
    }

    getImage(loc) {
        return super.getImage(Math.floor(loc[0]/2), Math.floor(loc[1]/2), 2*constants.TILEWIDTH, 2*constants.TILEHEIGHT);
    }

    getStartImage() {
        return this.getImage([8, 0])
    }

    getSprite() {
        return this.sprite;
    }

    drawPacman(x, y, dt) {
        let image;
        let stopimage;
        if (this.alive) {
            switch (this.direction) {
                case constants.LEFT:
                    image = this.getImage(this.animations.left.update(dt));
                    stopimage = [8, 0];
                    break;
                case constants.RIGHT:
                    image = this.getImage(this.animations.right.update(dt));
                    stopimage = [10, 0];
                    break;
                case constants.DOWN:
                    image = this.getImage(this.animations.down.update(dt));
                    stopimage = [8, 2]
                    break;
                case constants.UP:
                    image = this.getImage(this.animations.up.update(dt));
                    stopimage = [10, 2]
                    break;
                case constants.STOP:
                    image = this.getImage(this.stopimage);
                    
            }
        } else {
            image = this.getImage(this.animations.death);
        }

        this.sprite.x = (x + 0.5) * constants.TILEWIDTH;
        this.sprite.y = (y + 0.5) * constants.TILEHEIGHT;
        this.sprite.texture = image;
        
        // let pacmanSprite = new Sprite();
        // this.sprite = pacmanSprite;
        // pacmanSprite.anchor.set(0.5);
        // pacmanSprite.x = (col + 1/2) * constants.TILEWIDTH;
        // pacmanSprite.y = (row + 1/2) * constants.TILEHEIGHT;;
        // return pacmanSprite;
    }

    changeImage() {
        this.sprite.texture = this.getImage(this.animations.up.update(0));
    }

}

export default PacmanSprites;