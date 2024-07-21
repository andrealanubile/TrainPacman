import { Container, Graphics } from 'pixi.js';

class RewardBubble {
    constructor(type, init_x, init_y, acc, alpha_decay) {
        this.x = init_x;
        this.y = init_y;
        this.acc = acc;
        this.vel = 0;
        this.alpha_decay = alpha_decay;
        this.graphic = new Graphics();
        this.alpha = 1
        this.isAlive = true;
        switch (type) {
            case '+':
                this.graphic.circle(this.x, this.y, 6)
                this.graphic.fill(0xffffff);
                this.graphic.circle(this.x, this.y, 5);
                this.graphic.fill(0x04AA6D);
                break;
            case '++':
                this.graphic.circle(this.x, this.y, 12)
                this.graphic.fill(0xffffff);
                this.graphic.circle(this.x, this.y, 10);
                this.graphic.fill(0x04AA6D);
                break;
            case '-':
                this.graphic.circle(this.x, this.y, 6)
                this.graphic.fill(0xffffff);
                this.graphic.circle(this.x, this.y, 5);
                this.graphic.fill(0xab0404);
                break;
            case '--':
                this.graphic.circle(this.x, this.y, 12)
                this.graphic.fill(0xffffff);
                this.graphic.circle(this.x, this.y, 10);
                this.graphic.fill(0xab0404);
                break;
        }
    }

    getGraphic() {
        return this.graphic;
    }

    update(dt) {
        this.vel += this.acc*dt
        this.graphic.y -= this.vel*dt
        this.alpha -= this.alpha_decay*dt
        if (this.alpha >= 0) {
            this.graphic.alpha = this.alpha;
        } else {
            this.graphic.alpha = 0;
            this.isAlive = false;
        }


    }
}

export default RewardBubble;